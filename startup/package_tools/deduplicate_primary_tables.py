#!/usr/bin/env python3
"""Deduplicate worker-local TOPAS primary stopping-power or XS tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


SCHEMAS = {
    "stopping": {
        "input_columns": (
            "energy_MeV_per_u",
            "total_kinetic_energy_MeV",
            "electronic_stopping_power_MeV_per_mm",
            "total_stopping_power_MeV_per_mm",
            "csda_range_mm",
        ),
        "runtime_columns": (
            "energy_MeVu",
            "stopping_power_MeV_per_mm",
        ),
        "runtime_indices": (0, 2),
    },
    "cross-section": {
        "input_columns": (
            "energy_MeV_per_u",
            "total_kinetic_energy_MeV",
            "projectile_h_inelastic_cross_section_barn",
            "projectile_o_inelastic_cross_section_barn",
            "hydrogen_macroscopic_cross_section_per_mm",
            "oxygen_macroscopic_cross_section_per_mm",
            "water_macroscopic_cross_section_per_mm",
            "water_mean_free_path_mm",
        ),
        "runtime_columns": (
            "energy_MeV_per_u",
            "water_macroscopic_cross_section_per_mm",
        ),
        "runtime_indices": (0, 6),
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=tuple(SCHEMAS), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--header", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--diagnostic-output",
        type=Path,
        help="optional deduplicated CSV retaining every raw TOPAS column",
    )
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--relative-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--absolute-tolerance", type=float, default=1.0e-12)
    args = parser.parse_args()

    schema = SCHEMAS[args.kind]
    input_columns = schema["input_columns"]
    runtime_columns = schema["runtime_columns"]
    runtime_indices = schema["runtime_indices"]
    expected_width = len(input_columns)
    rows: dict[float, list[tuple[float, ...]]] = {}
    input_rows = 0
    with args.input.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            values = tuple(float(item) for item in line.split())
            if len(values) != expected_width or not all(map(math.isfinite, values)):
                raise SystemExit(f"Invalid {args.kind} row at line {line_number}")
            rows.setdefault(values[0], []).append(values)
            input_rows += 1
    if not rows:
        raise SystemExit(f"{args.input} contains no rows")

    duplicate_counts = {len(group) for group in rows.values()}
    if len(duplicate_counts) != 1:
        raise SystemExit(
            "Worker duplication count differs across energies: "
            + ", ".join(map(str, sorted(duplicate_counts)))
        )
    duplicate_count = next(iter(duplicate_counts))
    canonical: list[tuple[float, ...]] = []
    for energy, group in sorted(rows.items()):
        reference = group[0]
        for worker_index, candidate in enumerate(group[1:], start=1):
            for column, expected, actual in zip(input_columns, reference, candidate):
                if not math.isclose(
                    actual, expected,
                    rel_tol=args.relative_tolerance,
                    abs_tol=args.absolute_tolerance,
                ):
                    raise SystemExit(
                        f"Worker mismatch at {energy:g} MeV/u, {column}, "
                        f"copy {worker_index}: {actual:.17g} != {expected:.17g}"
                    )
        canonical.append(reference)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(runtime_columns)
        writer.writerows(
            tuple(row[index] for index in runtime_indices) for row in canonical
        )
    diagnostic = None
    if args.diagnostic_output is not None:
        args.diagnostic_output.parent.mkdir(parents=True, exist_ok=True)
        with args.diagnostic_output.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(input_columns)
            writer.writerows(canonical)
        diagnostic = {
            "path": args.diagnostic_output.as_posix(),
            "bytes": args.diagnostic_output.stat().st_size,
            "sha256": sha256(args.diagnostic_output),
            "rows": len(canonical),
            "columns": input_columns,
        }
    metadata = {
        "kind": args.kind,
        "input": {
            "path": args.input.as_posix(),
            "bytes": args.input.stat().st_size,
            "sha256": sha256(args.input),
            "header_path": args.header.as_posix(),
            "header_sha256": sha256(args.header),
            "rows": input_rows,
        },
        "worker_copies_per_energy": duplicate_count,
        "consistency_tolerance": {
            "relative": args.relative_tolerance,
            "absolute": args.absolute_tolerance,
        },
        "energy_range_MeV_per_u": {
            "minimum": canonical[0][0],
            "maximum": canonical[-1][0],
            "count": len(canonical),
        },
        "input_columns": input_columns,
        "runtime_columns": runtime_columns,
        "runtime_source_column_indices": runtime_indices,
        "output": {
            "path": args.output.as_posix(),
            "bytes": args.output.stat().st_size,
            "sha256": sha256(args.output),
            "rows": len(canonical),
            "columns": runtime_columns,
        },
    }
    if diagnostic is not None:
        metadata["diagnostic_output"] = diagnostic
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(
        f"Validated {duplicate_count} identical worker copies at "
        f"{len(canonical)} energies; wrote {args.output}"
    )


if __name__ == "__main__":
    main()
