#!/usr/bin/env python3
"""Compile validated TOPAS reaction CSV gzip files into a GPU-ready binary table."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import struct
from pathlib import Path


MAGIC = b"CRPKG01\0"
VERSION = 2
HEADER = struct.Struct("<8sIIIIIIffQQQ")
ENERGY_BIN = struct.Struct("<II")
REACTION = struct.Struct("<ffII")
SECONDARY = struct.Struct("<ihhffff")


def read_unit_direction(
    row: dict[str, str], columns: tuple[str, str, str], label: str
) -> tuple[float, float, float]:
    direction = tuple(float(row[column]) for column in columns)
    norm_squared = sum(component * component for component in direction)
    if (not all(math.isfinite(component) for component in direction)
            or abs(norm_squared - 1.0) > 2.0e-3):
        raise SystemExit(f"Invalid unit direction for {label}: {direction}")
    inverse_norm = 1.0 / math.sqrt(norm_squared)
    return tuple(component * inverse_norm for component in direction)


def direction_in_parent_frame(
    direction: tuple[float, float, float],
    parent: tuple[float, float, float],
) -> tuple[float, float, float]:
    wx, wy, wz = parent
    reference = (1.0, 0.0, 0.0) if abs(wx) < 0.9 else (0.0, 1.0, 0.0)
    projection = sum(reference[index] * parent[index] for index in range(3))
    u_raw = tuple(reference[index] - projection * parent[index] for index in range(3))
    u_norm = math.sqrt(sum(component * component for component in u_raw))
    u = tuple(component / u_norm for component in u_raw)
    v = (
        wy * u[2] - wz * u[1],
        wz * u[0] - wx * u[2],
        wx * u[1] - wy * u[0],
    )
    local = (
        sum(direction[index] * u[index] for index in range(3)),
        sum(direction[index] * v[index] for index in range(3)),
        sum(direction[index] * parent[index] for index in range(3)),
    )
    local_norm = math.sqrt(sum(component * component for component in local))
    return tuple(component / local_norm for component in local)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--reactions", type=Path, required=True)
    parser.add_argument("--secondaries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-metadata", type=Path, required=True)
    parser.add_argument("--energy-bin-min-mevu", type=float, default=0.0)
    parser.add_argument("--energy-bin-width-mevu", type=float, default=1.0)
    parser.add_argument("--energy-bin-count", type=int, default=201)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_gzip_csv(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def require_columns(rows: list[dict[str, str]], columns: tuple[str, ...], label: str) -> None:
    if not rows:
        raise SystemExit(f"{label} table is empty")
    missing = [column for column in columns if column not in rows[0]]
    if missing:
        raise SystemExit(f"{label} table is missing columns: {missing}")


def verify_source(path: Path, source: dict[str, object], rows: list[dict[str, str]]) -> None:
    expected_hash = str(source["sha256"])
    actual_hash = sha256(path)
    if actual_hash != expected_hash:
        raise SystemExit(f"SHA-256 mismatch for {path}: {actual_hash} != {expected_hash}")
    expected_rows = int(source["rows"])
    if len(rows) != expected_rows:
        raise SystemExit(f"Row-count mismatch for {path}: {len(rows)} != {expected_rows}")


def main() -> None:
    args = parse_arguments()
    if args.energy_bin_width_mevu <= 0.0 or args.energy_bin_count <= 0:
        raise SystemExit("Energy-bin width and count must be positive")

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    runtime = metadata.get("runtime_reference", {})
    if (runtime.get("topas_version") != "4.2.p3" or
            runtime.get("geant4_version") != "geant4-11-03-patch-02"):
        raise SystemExit(
            "Reaction packages require TOPAS 4.2.p3 / Geant4 11.3.2 provenance"
        )
    reaction_rows = read_gzip_csv(args.reactions)
    secondary_rows = read_gzip_csv(args.secondaries)
    require_columns(
        reaction_rows,
        (
            "reaction_id",
            "incident_c12_energy_MeV_per_u",
            "reaction_depth_mm",
            "secondary_count",
            "secondary_offset_zero_based",
            "incident_direction_x",
            "incident_direction_y",
            "incident_direction_z",
        ),
        "reaction",
    )
    require_columns(
        secondary_rows,
        (
            "reaction_id",
            "secondary_index",
            "pdg_id",
            "atomic_number_Z",
            "mass_number_A",
            "kinetic_energy_MeV",
            "direction_x",
            "direction_y",
            "direction_z",
        ),
        "secondary",
    )
    verify_source(args.reactions, metadata["output_files"]["reactions"], reaction_rows)
    verify_source(args.secondaries, metadata["output_files"]["secondaries"], secondary_rows)

    bins: list[list[int]] = [[] for _ in range(args.energy_bin_count)]
    expected_secondary_offset = 0
    for row_index, row in enumerate(reaction_rows):
        reaction_id = int(row["reaction_id"])
        if reaction_id != row_index + 1:
            raise SystemExit(f"Reaction IDs must be contiguous at row {row_index + 2}")
        offset = int(row["secondary_offset_zero_based"])
        count = int(row["secondary_count"])
        if offset != expected_secondary_offset:
            raise SystemExit(f"Non-contiguous secondary offset for reaction {reaction_id}")
        if offset + count > len(secondary_rows):
            raise SystemExit(f"Secondary range exceeds table for reaction {reaction_id}")
        for secondary_index, secondary in enumerate(secondary_rows[offset : offset + count], start=1):
            if int(secondary["reaction_id"]) != reaction_id:
                raise SystemExit(f"Secondary reaction ID mismatch for reaction {reaction_id}")
            if int(secondary["secondary_index"]) != secondary_index:
                raise SystemExit(f"Secondary indices are not contiguous for reaction {reaction_id}")
        expected_secondary_offset += count

        energy = float(row["incident_c12_energy_MeV_per_u"])
        bin_index = math.floor((energy - args.energy_bin_min_mevu) / args.energy_bin_width_mevu)
        if bin_index < 0 or bin_index >= args.energy_bin_count:
            raise SystemExit(f"Reaction {reaction_id} energy {energy} MeV/u is outside the bin grid")
        bins[bin_index].append(row_index)

    if expected_secondary_offset != len(secondary_rows):
        raise SystemExit("Reaction ranges do not consume the complete secondary table")
    empty_bins = [index for index, members in enumerate(bins) if not members]
    if empty_bins:
        raise SystemExit(
            "Every runtime energy bin needs at least one reaction package; empty bins: "
            + ", ".join(map(str, empty_bins[:20]))
        )

    ordered_reactions: list[tuple[float, float, int, int]] = []
    ordered_secondaries: list[tuple[int, int, int, float, float, float, float]] = []
    binary_bins: list[tuple[int, int]] = []
    for members in bins:
        binary_bins.append((len(ordered_reactions), len(members)))
        for row_index in members:
            row = reaction_rows[row_index]
            source_offset = int(row["secondary_offset_zero_based"])
            count = int(row["secondary_count"])
            incident_direction = read_unit_direction(
                row,
                ("incident_direction_x", "incident_direction_y", "incident_direction_z"),
                f"reaction {row['reaction_id']} incident particle",
            )
            output_offset = len(ordered_secondaries)
            ordered_reactions.append(
                (
                    float(row["incident_c12_energy_MeV_per_u"]),
                    float(row["reaction_depth_mm"]),
                    output_offset,
                    count,
                )
            )
            for secondary in secondary_rows[source_offset : source_offset + count]:
                atomic_number = int(secondary["atomic_number_Z"])
                mass_number = int(secondary["mass_number_A"])
                if not (-32768 <= atomic_number <= 32767 and -32768 <= mass_number <= 32767):
                    raise SystemExit("Secondary Z/A exceeds the binary int16 range")
                kinetic_energy = float(secondary["kinetic_energy_MeV"])
                if kinetic_energy < 0.0 or not math.isfinite(kinetic_energy):
                    raise SystemExit("Invalid secondary energy or direction")
                global_direction = read_unit_direction(
                    secondary, ("direction_x", "direction_y", "direction_z"),
                    f"reaction {row['reaction_id']} secondary",
                )
                local_direction = direction_in_parent_frame(
                    global_direction, incident_direction)
                pdg_id = int(secondary["pdg_id"])
                if secondary.get("particle_name") == "primary_continuation":
                    if not (
                        int(secondary["track_id"]) == 1
                        and pdg_id == 1_000_060_120
                        and atomic_number == 6
                        and mass_number == 12
                    ):
                        raise SystemExit(
                            "Invalid primary_continuation package member"
                        )
                    pdg_id = -pdg_id
                ordered_secondaries.append(
                    (
                        pdg_id,
                        atomic_number,
                        mass_number,
                        kinetic_energy,
                        *local_direction,
                    )
                )

    if len(ordered_reactions) > 0xFFFFFFFF or len(ordered_secondaries) > 0xFFFFFFFF:
        raise SystemExit("Version 2 binary offsets are limited to uint32")
    expected_file_size = (
        HEADER.size
        + len(binary_bins) * ENERGY_BIN.size
        + len(ordered_reactions) * REACTION.size
        + len(ordered_secondaries) * SECONDARY.size
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        stream.write(
            HEADER.pack(
                MAGIC,
                VERSION,
                HEADER.size,
                ENERGY_BIN.size,
                REACTION.size,
                SECONDARY.size,
                len(binary_bins),
                args.energy_bin_min_mevu,
                args.energy_bin_width_mevu,
                len(ordered_reactions),
                len(ordered_secondaries),
                expected_file_size,
            )
        )
        for record in binary_bins:
            stream.write(ENERGY_BIN.pack(*record))
        for record in ordered_reactions:
            stream.write(REACTION.pack(*record))
        for record in ordered_secondaries:
            stream.write(SECONDARY.pack(*record))

    if args.output.stat().st_size != expected_file_size:
        raise SystemExit("Compiled binary size does not match its header")
    compiled_metadata = {
        "format": "carbon reaction package binary",
        "format_version": VERSION,
        "byte_order": "little-endian",
        "direction_coordinates": "projectile-local orthonormal frame",
        "direction_components": ["local_x", "local_y", "along_projectile"],
        "source_metadata": args.metadata.as_posix(),
        "source_metadata_sha256": sha256(args.metadata),
        "source_runtime": {
            "topas_version": runtime.get("topas_version"),
            "geant4_version": runtime.get("geant4_version"),
            "runtime_reference_metadata": runtime.get("metadata"),
        },
        "source_reactions_sha256": sha256(args.reactions),
        "source_secondaries_sha256": sha256(args.secondaries),
        "energy_bins": {
            "minimum_MeV_per_u": args.energy_bin_min_mevu,
            "width_MeV_per_u": args.energy_bin_width_mevu,
            "count": len(binary_bins),
            "minimum_packages_per_bin": min(map(len, bins)),
            "maximum_packages_per_bin": max(map(len, bins)),
        },
        "records": {
            "reactions": len(ordered_reactions),
            "secondaries": len(ordered_secondaries),
        },
        "record_sizes_bytes": {
            "header": HEADER.size,
            "energy_bin": ENERGY_BIN.size,
            "reaction": REACTION.size,
            "secondary": SECONDARY.size,
        },
        "output": {
            "path": args.output.as_posix(),
            "bytes": args.output.stat().st_size,
            "sha256": sha256(args.output),
        },
    }
    args.output_metadata.parent.mkdir(parents=True, exist_ok=True)
    with args.output_metadata.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(compiled_metadata, indent=2) + "\n")
    print(f"Wrote {args.output} ({expected_file_size} bytes)")
    print(f"Wrote {args.output_metadata}")
    print(
        f"Compiled {len(ordered_reactions)} reactions and "
        f"{len(ordered_secondaries)} secondaries into {len(binary_bins)} energy bins"
    )


if __name__ == "__main__":
    main()
