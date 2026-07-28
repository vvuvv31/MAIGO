#!/usr/bin/env python3
"""Merge prepared reaction/secondary CSV pairs into one validated source."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing CSV header")
        return list(reader.fieldnames), list(reader)


def write_csv(
    path: Path, fieldnames: list[str], rows: list[dict[str, str]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(
        path, "wt", encoding="utf-8", newline="", compresslevel=9
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        nargs=3,
        metavar=("METADATA", "REACTIONS", "SECONDARIES"),
        required=True,
    )
    parser.add_argument("--reactions-output", type=Path, required=True)
    parser.add_argument("--secondaries-output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    args = parser.parse_args()

    merged_reactions: list[dict[str, str]] = []
    merged_secondaries: list[dict[str, str]] = []
    reaction_fields: list[str] | None = None
    secondary_fields: list[str] | None = None
    source_records: list[dict[str, object]] = []
    next_reaction_id = 1
    for metadata_name, reaction_name, secondary_name in args.source:
        metadata_path = Path(metadata_name)
        reaction_path = Path(reaction_name)
        secondary_path = Path(secondary_name)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        fields_r, reactions = read_csv(reaction_path)
        fields_s, secondaries = read_csv(secondary_path)
        if reaction_fields is None:
            reaction_fields = fields_r
            secondary_fields = fields_s
        elif fields_r != reaction_fields or fields_s != secondary_fields:
            raise ValueError("source CSV schemas do not match")

        secondary_by_reaction: dict[int, list[dict[str, str]]] = {}
        for row in secondaries:
            secondary_by_reaction.setdefault(
                int(row["reaction_id"]), []
            ).append(row)
        for source_reaction in reactions:
            old_id = int(source_reaction["reaction_id"])
            products = secondary_by_reaction.get(old_id, [])
            expected = int(source_reaction["secondary_count"])
            if len(products) != expected:
                raise ValueError(
                    f"{reaction_path}: reaction {old_id} has "
                    f"{len(products)} products, expected {expected}"
                )
            reaction = dict(source_reaction)
            reaction["reaction_id"] = str(next_reaction_id)
            reaction["secondary_offset_zero_based"] = str(
                len(merged_secondaries)
            )
            merged_reactions.append(reaction)
            for index, source_secondary in enumerate(products, start=1):
                secondary = dict(source_secondary)
                secondary["reaction_id"] = str(next_reaction_id)
                secondary["secondary_index"] = str(index)
                merged_secondaries.append(secondary)
            next_reaction_id += 1
        source_records.append(
            {
                "metadata": metadata_path.as_posix(),
                "metadata_sha256": sha256(metadata_path),
                "reactions": reaction_path.as_posix(),
                "reactions_sha256": sha256(reaction_path),
                "secondaries": secondary_path.as_posix(),
                "secondaries_sha256": sha256(secondary_path),
                "reaction_rows": len(reactions),
                "secondary_rows": len(secondaries),
                "source_metadata": metadata,
            }
        )

    assert reaction_fields is not None and secondary_fields is not None
    write_csv(args.reactions_output, reaction_fields, merged_reactions)
    write_csv(args.secondaries_output, secondary_fields, merged_secondaries)
    output_metadata = {
        "format": "merged_correlated_reaction_package_source_v1",
        "sources": source_records,
        "output_files": {
            "reactions": {
                "path": args.reactions_output.as_posix(),
                "rows": len(merged_reactions),
                "sha256": sha256(args.reactions_output),
            },
            "secondaries": {
                "path": args.secondaries_output.as_posix(),
                "rows": len(merged_secondaries),
                "sha256": sha256(args.secondaries_output),
            },
        },
    }
    args.metadata_output.write_text(
        json.dumps(output_metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(args.reactions_output)
    print(args.secondaries_output)
    print(args.metadata_output)
    print(
        f"merged reactions={len(merged_reactions)} "
        f"secondaries={len(merged_secondaries)}"
    )


if __name__ == "__main__":
    main()
