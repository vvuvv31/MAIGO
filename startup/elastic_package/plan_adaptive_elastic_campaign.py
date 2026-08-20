#!/usr/bin/env python3
"""Plan fixed-energy elastic sampling runs for sparse 1 MeV/u bins.

The planner only writes an auditable JSON manifest.  It does not launch TOPAS.
Fixed-energy runs are package-sampling extractions: continuous EM loss and
multiple scattering are disabled, and the event scorer accepts only hadElastic.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


DIRECT_FIELDS = [
    "energy_MeV_per_u",
    "water_macroscopic_cross_section_per_mm",
]
DIRECT_SOURCE = "direct G4HadronicProcessStore query"
MAX_DIRECT_POINTS = 1_000_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("must be finite")
    return parsed


def energy_tag(energy: float) -> str:
    return f"{energy:g}".replace(".", "p")


def projectile_tag(name: str, atomic_number: int, mass_number: int) -> str:
    if name == "proton" and (atomic_number, mass_number) == (1, 1):
        return "proton"
    return f"ion_z{atomic_number}a{mass_number}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--direct-xs-csv", type=Path, required=True)
    parser.add_argument("--direct-xs-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-events", type=positive_int, default=1000)
    parser.add_argument("--path-length-mm", type=positive_float, default=700.0)
    parser.add_argument("--margin", type=positive_float, default=1.25)
    parser.add_argument("--min-histories", type=positive_int, default=100)
    parser.add_argument("--max-histories", type=positive_int, default=10_000_000)
    parser.add_argument("--energy-min-mevu", type=finite_float, default=0.0)
    parser.add_argument("--energy-max-mevu", type=finite_float, default=250.0)
    parser.add_argument("--projectile-z", type=positive_int, default=1)
    parser.add_argument("--projectile-a", type=positive_int, default=1)
    parser.add_argument("--projectile-name", default="proton")
    parser.add_argument("--material", default="G4_WATER")
    parser.add_argument("--physics-model", default="G4HadronElasticPhysicsHP")
    parser.add_argument("--topas-version", default=None)
    parser.add_argument("--geant4-version", default=None)
    parser.add_argument("--remote-root", default="/home/v/maigo_elastic_adaptive_1000")
    parser.add_argument("--seed-base", type=nonnegative_int, default=2026086000)
    parser.add_argument("--first-batch-size", type=nonnegative_int, default=16)
    args = parser.parse_args()
    if args.projectile_a < args.projectile_z:
        parser.error("projectile A must be >= Z")
    if args.max_histories < args.min_histories:
        parser.error("max-histories must be >= min-histories")
    span = args.energy_max_mevu - args.energy_min_mevu
    if args.energy_min_mevu < 0.0 or span < 0.0 or abs(span - round(span)) > 1.0e-9:
        parser.error("energy bounds must define integer-aligned 1 MeV/u bins")
    return args


def load_direct_table(csv_path: Path, metadata_path: Path, args: argparse.Namespace) -> tuple[dict[str, object], list[tuple[float, float]]]:
    if not csv_path.is_file():
        raise SystemExit(f"Missing direct-query CSV: {csv_path}")
    if not metadata_path.is_file():
        raise SystemExit(f"Missing direct-query metadata: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"Invalid direct-query metadata: {metadata_path}") from error
    if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
        raise SystemExit(f"{metadata_path}: unsupported direct-query metadata schema")
    if metadata.get("kind") != "direct_elastic_cross_section_query":
        raise SystemExit(f"{metadata_path}: invalid direct-query metadata kind")
    if metadata.get("source") != DIRECT_SOURCE:
        raise SystemExit(f"{metadata_path}: invalid direct-query source")
    projectile = metadata.get("projectile")
    if not isinstance(projectile, dict) or projectile.get("Z") != args.projectile_z or projectile.get("A") != args.projectile_a:
        raise SystemExit(f"{metadata_path}: projectile identity differs from planner")
    if metadata.get("material") != args.material or metadata.get("process") != "hadElastic":
        raise SystemExit(f"{metadata_path}: material/process identity mismatch")
    model = metadata.get("physics_model")
    if model != args.physics_model:
        raise SystemExit(f"{metadata_path}: physics model differs from planner")
    grid = metadata.get("grid")
    if not isinstance(grid, dict):
        raise SystemExit(f"{metadata_path}: missing grid metadata")
    try:
        minimum = float(grid["minimum_MeV_per_u"])
        maximum = float(grid["maximum_MeV_per_u"])
        step = float(grid["step_MeV_per_u"])
        points = grid["points"]
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"{metadata_path}: incomplete grid metadata") from error
    expected_points = int(round((maximum - minimum) / step)) + 1 if step > 0.0 else 0
    if (
        not all(math.isfinite(value) for value in (minimum, maximum, step))
        or minimum < 0.0 or maximum < minimum or step <= 0.0
        or not isinstance(points, int) or isinstance(points, bool)
        or points < 2 or points > MAX_DIRECT_POINTS or points != expected_points
        or grid.get("endpoint_inclusive") is not True
    ):
        raise SystemExit(f"{metadata_path}: invalid direct-query grid")
    output = metadata.get("output")
    if not isinstance(output, dict) or not isinstance(output.get("path"), str) or not isinstance(output.get("sha256"), str):
        raise SystemExit(f"{metadata_path}: missing direct-query output provenance")
    recorded = Path(output["path"])
    candidates = (recorded, metadata_path.parent / recorded, metadata_path.parent / recorded.name)
    resolved = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
    if resolved != csv_path.resolve():
        raise SystemExit(f"{metadata_path}: metadata output path does not match direct-query CSV")
    if output["sha256"] != sha256(csv_path):
        raise SystemExit(f"{metadata_path}: SHA-256 mismatch for direct-query CSV")
    if minimum > args.energy_min_mevu or maximum < args.energy_max_mevu + 1.0:
        raise SystemExit(f"{metadata_path}: direct-query grid does not cover requested bins")
    rows: list[tuple[float, float]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != DIRECT_FIELDS:
            raise SystemExit(f"{csv_path}: unexpected direct-query CSV header")
        for index, row in enumerate(reader, 2):
            try:
                energy = float(row[DIRECT_FIELDS[0]])
                value = float(row[DIRECT_FIELDS[1]])
            except (KeyError, TypeError, ValueError) as error:
                raise SystemExit(f"{csv_path}: invalid row {index}") from error
            if not math.isfinite(energy) or energy < 0.0 or not math.isfinite(value) or value < 0.0:
                raise SystemExit(f"{csv_path}: non-finite or negative row {index}")
            expected = minimum + len(rows) * step
            if abs(energy - expected) > 1.0e-9:
                raise SystemExit(f"{csv_path}: nonuniform direct-query grid at row {index}")
            rows.append((energy, value))
    if len(rows) != points or abs(rows[-1][0] - maximum) > 1.0e-9:
        raise SystemExit(f"{csv_path}: point count or endpoint does not match metadata")
    return metadata, rows


def interpolate(rows: list[tuple[float, float]], energy: float) -> float:
    if energy < rows[0][0] or energy > rows[-1][0]:
        raise SystemExit(f"Representative energy {energy:g} is outside direct-query grid")
    position = (energy - rows[0][0]) / (rows[1][0] - rows[0][0])
    index = int(math.floor(position))
    index = max(0, min(index, len(rows) - 2))
    left_energy, left_value = rows[index]
    right_energy, right_value = rows[index + 1]
    fraction = (energy - left_energy) / (right_energy - left_energy)
    return left_value + fraction * (right_value - left_value)


def representative_energy(bin_minimum: float, rows: list[tuple[float, float]]) -> float:
    if abs(bin_minimum) <= 1.0e-12:
        preferred = 0.9
        if rows[0][0] <= preferred <= rows[-1][0] and interpolate(rows, preferred) > 0.0:
            return preferred
        for energy, value in rows:
            if 0.0 <= energy < 1.0 and value > 0.0:
                return energy
        raise SystemExit("Bin 0 has no positive direct elastic XS representative")
    return bin_minimum + 0.5


def load_coverage(path: Path, minimum: float, maximum: float) -> list[int]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"Invalid coverage JSON: {path}") from error
    bins = document.get("bins") if isinstance(document, dict) else None
    expected = int(round(maximum - minimum)) + 1
    if not isinstance(bins, list) or len(bins) != expected:
        raise SystemExit(f"{path}: coverage must contain exactly {expected} bins")
    counts: list[int] = []
    for index, entry in enumerate(bins):
        if not isinstance(entry, dict) or entry.get("index") != index:
            raise SystemExit(f"{path}: coverage bin index {index} is invalid")
        count = entry.get("event_count")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise SystemExit(f"{path}: coverage event_count for bin {index} is invalid")
        counts.append(count)
    return counts


def main() -> None:
    args = parse_args()
    metadata, rows = load_direct_table(args.direct_xs_csv.resolve(), args.direct_xs_metadata.resolve(), args)
    counts = load_coverage(args.coverage.resolve(), args.energy_min_mevu, args.energy_max_mevu)
    bin_count = len(counts)
    projectile_label = projectile_tag(args.projectile_name, args.projectile_z, args.projectile_a)
    entries: list[dict[str, object]] = []
    planned_indices: list[int] = []
    for index, current in enumerate(counts):
        deficit = max(0, args.target_events - current)
        bin_minimum = args.energy_min_mevu + index
        energy = representative_energy(bin_minimum, rows)
        xs = interpolate(rows, energy)
        if deficit == 0:
            entries.append({
                "bin": index, "minimum_MeV_per_u": bin_minimum,
                "maximum_MeV_per_u": bin_minimum + 1.0,
                "target_events": args.target_events, "current_events": current,
                "deficit_events": 0, "representative_energy_MeV_per_u": energy,
                "direct_xs_per_mm": xs, "event_probability": None,
                "histories": 0, "seed": None, "label": None,
                "status": "skipped_satisfied",
            })
            continue
        if not math.isfinite(xs) or xs <= 0.0:
            raise SystemExit(f"Bin {index} representative energy {energy:g} has zero or invalid direct XS")
        optical_depth = xs * args.path_length_mm
        probability = -math.expm1(-optical_depth)
        if not math.isfinite(probability) or probability <= 0.0 or probability > 1.0:
            raise SystemExit(f"Bin {index} has invalid event probability {probability!r}")
        histories = max(args.min_histories, math.ceil(deficit * args.margin / probability))
        if histories > args.max_histories:
            raise SystemExit(f"Bin {index} requires {histories} histories, above max-histories {args.max_histories}")
        label = f"{projectile_label}_{energy_tag(energy)}mevu_g4_water"
        entries.append({
            "bin": index, "minimum_MeV_per_u": bin_minimum,
            "maximum_MeV_per_u": bin_minimum + 1.0,
            "target_events": args.target_events, "current_events": current,
            "deficit_events": deficit, "representative_energy_MeV_per_u": energy,
            "direct_xs_per_mm": xs, "path_length_mm": args.path_length_mm,
            "optical_depth": optical_depth, "event_probability": probability,
            "margin": args.margin, "histories": histories,
            "seed": args.seed_base + index, "label": label,
            "status": "planned",
            "output": {
                "remote_root": args.remote_root,
                "run_dir": f"{args.remote_root}/runs/{label}",
                "prepared_metadata": f"{args.remote_root}/runs/{label}/{label}.metadata.json",
                "interactions": f"{args.remote_root}/runs/{label}/{label}_interactions.csv.gz",
                "products": f"{args.remote_root}/runs/{label}/{label}_products.csv.gz",
            },
        })
        planned_indices.append(index)
    priority = sorted(planned_indices, key=lambda index: (counts[index], -index))
    first_batch = set(priority[:args.first_batch_size])
    for entry in entries:
        if entry["status"] == "planned":
            entry["batch"] = 1 if entry["bin"] in first_batch else 2
    first_batch_entries = [
        {
            key: entries[index][key]
            for key in (
                "bin", "current_events", "deficit_events",
                "representative_energy_MeV_per_u", "direct_xs_per_mm",
                "event_probability", "histories", "seed", "label",
            )
        }
        for index in sorted(first_batch)
    ]
    versions = metadata.get("versions") if isinstance(metadata.get("versions"), dict) else {}
    document = {
        "schema_version": 1,
        "kind": "adaptive_elastic_campaign_plan",
        "planner": {"name": "plan_adaptive_elastic_campaign.py", "version": 1},
        "identity": {
            "projectile": {"name": args.projectile_name, "Z": args.projectile_z, "A": args.projectile_a},
            "material": args.material, "process": "hadElastic", "physics_model": args.physics_model,
            "TOPAS": args.topas_version if args.topas_version is not None else versions.get("TOPAS"),
            "Geant4": args.geant4_version if args.geant4_version is not None else versions.get("Geant4"),
            "physics_modules": ["g4h-phy_QGSP_BIC_HP", "g4h-elastic_HP"],
        },
        "provenance": {
            "coverage": {"path": str(args.coverage.resolve()), "sha256": sha256(args.coverage.resolve())},
            "direct_xs": {"path": str(args.direct_xs_csv.resolve()), "sha256": sha256(args.direct_xs_csv.resolve()),
                          "metadata_path": str(args.direct_xs_metadata.resolve()), "metadata_sha256": sha256(args.direct_xs_metadata.resolve())},
            "extraction_mode": "fixed-energy-package-sampling",
            "sampling_purpose": "elastic-package-fixed-energy-sampling-not-dose-reference",
            "continuous_em_loss_enabled": False, "multiple_scattering_enabled": False,
            "scored_processes": ["hadElastic"], "phantom_half_length_mm": args.path_length_mm / 2.0,
        },
        "formula": "P = 1 - exp(-Sigma_direct_per_mm * L_mm); histories = ceil(deficit_events * margin / P)",
        "parameters": {
            "target_events_per_bin": args.target_events, "energy_grid": {"minimum_MeV_per_u": args.energy_min_mevu, "maximum_MeV_per_u": args.energy_max_mevu, "width_MeV_per_u": 1.0},
            "path_length_mm": args.path_length_mm, "margin": args.margin,
            "min_histories": args.min_histories, "max_histories": args.max_histories,
            "seed_base": args.seed_base, "first_batch_size": args.first_batch_size,
        },
        "summary": {
            "bins": bin_count, "planned_bins": len(planned_indices), "skipped_bins": bin_count - len(planned_indices),
            "planned_histories": sum(int(entry["histories"]) for entry in entries),
            "first_batch_bins": sorted(first_batch),
            "first_batch_entries": first_batch_entries,
        },
        "entries": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}: bins={bin_count} planned={len(planned_indices)} skipped={bin_count-len(planned_indices)} histories={document['summary']['planned_histories']}")
    print(f"first batch bins: {sorted(first_batch)}")


if __name__ == "__main__":
    main()
