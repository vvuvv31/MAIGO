#!/usr/bin/env python3
"""Convert a TOPAS primary-C12 reaction n-tuple into joint sampling records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_log(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="replace")
    result: dict[str, object] = {"path": path.as_posix(), "sha256": sha256(path)}
    topas_match = re.search(r"Welcome to TOPAS.*?Version\s+([^\)]+)\)", text)
    geant4_match = re.search(r"Geant4 version Name:\s+(\S+)", text)
    elapsed_match = re.search(
        r"^\s*Total:\s+User=[^\n]*?Real=([0-9.]+)s", text, flags=re.MULTILINE
    )
    if topas_match:
        result["topas_version"] = topas_match.group(1).strip()
    if geant4_match:
        result["geant4_version"] = geant4_match.group(1)
    if elapsed_match:
        result["elapsed_real_s"] = float(elapsed_match.group(1))
    return result


def parse_row(raw_line: str, path: Path, line_number: int) -> dict[str, object]:
    values = raw_line.split()
    if len(values) != 23:
        raise ValueError(f"Expected 23 columns at {path}:{line_number}, got {len(values)}")
    integer_indices = (1, 2, 3, 4, 5, 7, 8, 20, 21, 22)
    float_indices = (9, 10, 11, 12, 13, 14, 15, 16, 17)
    for index in integer_indices:
        int(values[index])
    for index in float_indices:
        value = float(values[index])
        if not math.isfinite(value):
            raise ValueError(f"Non-finite value at {path}:{line_number}")
    return {
        "record_kind": values[0],
        "run_id": int(values[1]),
        "event_id": int(values[2]),
        "track_id": int(values[3]),
        "parent_id": int(values[4]),
        "pdg_id": int(values[5]),
        "particle_name": values[6],
        "atomic_number": int(values[7]),
        "mass_number": int(values[8]),
        "charge_e": float(values[9]),
        "kinetic_energy_mev": float(values[10]),
        "incident_energy_mev": float(values[11]),
        "vertex_x_mm": float(values[12]),
        "vertex_y_mm": float(values[13]),
        "vertex_z_mm": float(values[14]),
        "direction_x": float(values[15]),
        "direction_y": float(values[16]),
        "direction_z": float(values[17]),
        "weight": float(values[18]),
        "process_name": values[19],
        "process_type": int(values[20]),
        "process_subtype": int(values[21]),
        "creator_model_id": int(values[22]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("smoke", "development", "reference"), required=True)
    parser.add_argument("--histories", type=int, required=True)
    parser.add_argument("--input-dir", type=Path, default=Path("validation/topas/output"))
    parser.add_argument("--phantom-half-length-mm", type=float, default=200.0)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--header", type=Path)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()
    if args.histories <= 0:
        raise SystemExit("--histories must be positive")

    stem = args.input_dir / f"fragment_{args.case}_reaction_vertices"
    input_path = args.input or stem.with_suffix(".phsp")
    header_path = args.header or stem.with_suffix(".header")
    log_path = args.log or args.input_dir / f"fragment-{args.case}_topas.log"
    for path in (input_path, header_path, log_path):
        if not path.exists():
            raise SystemExit(f"Required TOPAS output not found: {path}")

    records: list[dict[str, object]] = []
    with input_path.open(encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            stripped = raw_line.strip()
            if stripped:
                records.append(parse_row(stripped, input_path, line_number))

    reactions: dict[tuple[int, int], dict[str, object]] = {}
    secondaries: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        key = (int(record["run_id"]), int(record["event_id"]))
        if record["record_kind"] == "reaction":
            if key in reactions:
                raise SystemExit(f"More than one primary reaction header for run/event {key}")
            reactions[key] = record
        elif record["record_kind"] == "secondary":
            secondaries[key].append(record)
        else:
            raise SystemExit(f"Unknown record kind: {record['record_kind']}")

    orphan_keys = sorted(set(secondaries) - set(reactions))
    empty_keys = sorted(set(reactions) - set(secondaries))
    if orphan_keys:
        raise SystemExit(f"Secondary records without reaction headers: {orphan_keys[:5]}")
    if empty_keys:
        raise SystemExit(f"Reaction headers without secondary records: {empty_keys[:5]}")

    output_rows: list[list[object]] = []
    species_counts: Counter[str] = Counter()
    vertex_tolerance_mm = 2.0e-4
    for reaction_index, key in enumerate(sorted(reactions), start=1):
        reaction = reactions[key]
        incident_energy = float(reaction["incident_energy_mev"])
        for secondary_index, secondary in enumerate(secondaries[key], start=1):
            separation = max(
                abs(float(secondary[coordinate]) - float(reaction[coordinate]))
                for coordinate in ("vertex_x_mm", "vertex_y_mm", "vertex_z_mm")
            )
            if separation > vertex_tolerance_mm:
                raise SystemExit(f"Reaction/secondary vertex mismatch for run/event {key}")
            raw_incident = float(secondary["incident_energy_mev"])
            if raw_incident != 0.0 and abs(raw_incident - incident_energy) > 2.0e-3:
                raise SystemExit(f"Incident-energy mismatch for run/event {key}")

            mass_number = int(secondary["mass_number"])
            particle_energy = float(secondary["kinetic_energy_mev"])
            species_counts[str(secondary["particle_name"])] += 1
            output_rows.append(
                [
                    reaction_index,
                    secondary_index,
                    key[0],
                    key[1],
                    secondary["track_id"],
                    secondary["pdg_id"],
                    secondary["particle_name"],
                    secondary["atomic_number"],
                    mass_number,
                    secondary["charge_e"],
                    particle_energy,
                    particle_energy / mass_number if mass_number > 0 else "",
                    incident_energy,
                    incident_energy / 12.0,
                    float(reaction["vertex_z_mm"]) + args.phantom_half_length_mm,
                    reaction["vertex_x_mm"],
                    reaction["vertex_y_mm"],
                    reaction["vertex_z_mm"],
                    secondary["direction_x"],
                    secondary["direction_y"],
                    secondary["direction_z"],
                    secondary["weight"],
                    secondary["process_name"],
                    secondary["process_subtype"],
                    secondary["creator_model_id"],
                ]
            )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "reaction_id",
                "secondary_index",
                "run_id",
                "event_id",
                "track_id",
                "pdg_id",
                "particle_name",
                "atomic_number_Z",
                "mass_number_A",
                "charge_e",
                "kinetic_energy_MeV",
                "kinetic_energy_MeV_per_u",
                "incident_c12_energy_MeV",
                "incident_c12_energy_MeV_per_u",
                "reaction_depth_mm",
                "vertex_x_mm",
                "vertex_y_mm",
                "vertex_z_global_mm",
                "direction_x",
                "direction_y",
                "direction_z",
                "weight",
                "creator_process",
                "creator_process_subtype",
                "creator_model_id",
            )
        )
        for row in output_rows:
            writer.writerow(f"{value:.12g}" if isinstance(value, float) else value for value in row)

    multiplicities = [len(secondaries[key]) for key in reactions]
    depths = [
        float(reaction["vertex_z_mm"]) + args.phantom_half_length_mm
        for reaction in reactions.values()
    ]
    incident_energies_per_u = [
        float(reaction["incident_energy_mev"]) / 12.0 for reaction in reactions.values()
    ]
    metadata = {
        "case": args.case,
        "histories": args.histories,
        "reaction_definition": "first primary C-12 ionInelastic final state",
        "sampling_semantics": (
            "Rows sharing reaction_id are one correlated final-state package and must be sampled jointly"
        ),
        "coordinate_semantics": (
            f"reaction_depth_mm = TOPAS global vertex_z + {args.phantom_half_length_mm:g} mm"
        ),
        "reaction_count": len(reactions),
        "secondary_count": len(output_rows),
        "fraction_of_histories_with_primary_inelastic_reaction": len(reactions) / args.histories,
        "secondary_multiplicity": {
            "minimum": min(multiplicities) if multiplicities else 0,
            "maximum": max(multiplicities) if multiplicities else 0,
            "mean": statistics.fmean(multiplicities) if multiplicities else 0.0,
        },
        "reaction_depth_mm": {
            "minimum": min(depths) if depths else None,
            "maximum": max(depths) if depths else None,
        },
        "incident_c12_energy_MeV_per_u": {
            "minimum": min(incident_energies_per_u) if incident_energies_per_u else None,
            "maximum": max(incident_energies_per_u) if incident_energies_per_u else None,
        },
        "particle_counts": dict(sorted(species_counts.items())),
        "input_files": {
            "ntuple": {"path": input_path.as_posix(), "sha256": sha256(input_path)},
            "header": {"path": header_path.as_posix(), "sha256": sha256(header_path)},
            "topas_log": parse_log(log_path),
        },
        "output_csv": args.output_csv.as_posix(),
    }
    with args.metadata.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(metadata, indent=2) + "\n")

    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.metadata}")
    print(
        f"Primary reactions: {len(reactions)}/{args.histories}; "
        f"secondaries: {len(output_rows)}; mean multiplicity: {metadata['secondary_multiplicity']['mean']:.3f}"
    )


if __name__ == "__main__":
    main()
