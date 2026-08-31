#!/usr/bin/env python3
"""Aggregate complete TOPAS primary-path exposure ntuples.

``CarbonInelasticExposureNtuple`` writes one row per target-isotope path
segment, collision, and terminal history outcome.  This tool validates that
contract, splits nothing implicitly, rejects duplicate event identities, and
produces the CSV consumed by ``compile_cinel02_rates.py``.  The compiler then
adds the macroscopic rate, confidence interval, coverage qualification, and
runtime-table provenance.

The exposure denominator is the weighted *track length* in each target/energy
cell.  A collision row contributes only its authoritative target isotope and
authoritative collision energy.  History outcomes are cause-specific: a
history is censored for target ``t`` when it touched that cell but never had a
collision on target ``t`` (including when it collided on a competing target).
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


EXPOSURE_COLUMNS = (
    "record_kind",
    "run_id",
    "thread_id",
    "event_id",
    "track_id",
    "projectile_z",
    "projectile_a",
    "target_z",
    "target_a",
    "target_isotope_id",
    "target_isotope_name",
    "material_name",
    "energy_bin_id",
    "energy_low_MeV_per_u",
    "energy_high_MeV_per_u",
    "energy_MeV_per_u",
    "collision_energy_MeV_per_u",
    "track_length_mm",
    "track_weight",
    "weighted_track_length_mm",
    "target_number_density_per_mm3",
    "target_areal_density_weighted_per_mm2",
    "history_count",
    "history_weight_sum",
    "history_weight_squared_sum",
    "censored_history_count",
    "censored_history_weight_sum",
    "censored_history_weight_squared_sum",
    "collision_count",
    "collision_weight_sum",
    "collision_weight_squared_sum",
    "source_energy_MeV_per_u",
    "step_index",
    "collision_sequence",
    "collision_process",
    "collision_energy_source",
)


RATE_INPUT_COLUMNS = (
    "projectile_z",
    "projectile_a",
    "target_z",
    "target_a",
    "energy_bin_id",
    "energy_low_MeV_per_u",
    "energy_high_MeV_per_u",
    "energy_MeV_per_u",
    "weighted_track_length_mm",
    "target_areal_density_weighted_per_mm2",
    "target_number_density_per_mm3",
    "collision_count",
    "sum_weight",
    "sum_weight_squared",
    "history_count",
    "censored_history_count",
    "history_weight_sum",
    "history_weight_squared_sum",
    "censored_history_weight_sum",
    "censored_history_weight_squared_sum",
)


_COUNT_PATTERNS = {
    "original_histories": re.compile(r"^\s*Number of Original Histories:\s*(\d+)\s*$"),
    "scored_entries": re.compile(r"^\s*Number of Scored Entries:\s*(\d+)\s*$"),
}
_COLUMN_PATTERN = re.compile(r"^\s*(\d+):\s*(.*?)\s*$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_topas_paths(path: Path) -> tuple[Path, Path]:
    if path.suffix == ".header":
        return path, path.with_suffix(".phsp")
    if path.suffix == ".phsp":
        return path.with_suffix(".header"), path
    return Path(f"{path}.header"), Path(f"{path}.phsp")


def _read_topas_header(path: Path) -> dict[str, object]:
    counts: dict[str, int] = {}
    indexed_columns: list[tuple[int, str]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        matched_count = False
        for key, pattern in _COUNT_PATTERNS.items():
            match = pattern.match(line)
            if match:
                if key in counts:
                    raise ValueError(f"{path}:{line_number}: duplicate {key}")
                counts[key] = int(match.group(1))
                matched_count = True
                break
        if matched_count:
            continue
        match = _COLUMN_PATTERN.match(line)
        if match:
            indexed_columns.append((int(match.group(1)), match.group(2)))
    missing = [name for name in _COUNT_PATTERNS if name not in counts]
    if missing:
        raise ValueError(f"{path}: missing header fields: {', '.join(missing)}")
    indices = [index for index, _ in indexed_columns]
    if indices != list(range(1, len(indices) + 1)):
        raise ValueError(f"{path}: column indices are not contiguous: {indices}")
    columns = tuple(name for _, name in indexed_columns)
    if columns != EXPOSURE_COLUMNS:
        raise ValueError(
            f"{path}: exposure scorer schema mismatch; expected {EXPOSURE_COLUMNS}, "
            f"got {columns}"
        )
    return {
        "original_histories": counts["original_histories"],
        "scored_entries": counts["scored_entries"],
        "columns": columns,
    }


def _parse_topas(path: Path) -> tuple[list[dict[str, str]], dict[str, object], list[dict[str, str]]]:
    header_path, phsp_path = _resolve_topas_paths(path)
    if not header_path.is_file() or not phsp_path.is_file():
        raise FileNotFoundError(
            f"TOPAS exposure input requires {header_path} and {phsp_path}"
        )
    header = _read_topas_header(header_path)
    rows: list[dict[str, str]] = []
    with phsp_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            values = line.split()
            if len(values) != len(EXPOSURE_COLUMNS):
                raise ValueError(
                    f"{phsp_path}:{line_number}: expected {len(EXPOSURE_COLUMNS)} "
                    f"fields, got {len(values)}"
                )
            rows.append(dict(zip(EXPOSURE_COLUMNS, values)))
    if len(rows) != int(header["scored_entries"]):
        raise ValueError(
            f"{phsp_path}: header declares {header['scored_entries']} scored entries, "
            f"but parsed {len(rows)}"
        )
    files = [
        {"path": str(header_path), "sha256": _sha256(header_path), "kind": "header"},
        {"path": str(phsp_path), "sha256": _sha256(phsp_path), "kind": "phsp"},
    ]
    return rows, header, files


def _parse_csv(path: Path) -> tuple[list[dict[str, str]], dict[str, object], list[dict[str, str]]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing CSV header")
        if tuple(reader.fieldnames) != EXPOSURE_COLUMNS:
            raise ValueError(
                f"{path}: exposure CSV schema mismatch; expected {EXPOSURE_COLUMNS}, "
                f"got {tuple(reader.fieldnames)}"
            )
        rows = [{key: (value or "").strip() for key, value in row.items()} for row in reader]
    return rows, {"original_histories": None, "scored_entries": len(rows), "columns": EXPOSURE_COLUMNS}, [
        {"path": str(path), "sha256": _sha256(path), "kind": "csv"}
    ]


def read_input(path: Path) -> tuple[list[dict[str, str]], dict[str, object], list[dict[str, str]]]:
    if path.suffix in {".csv", ".gz"}:
        return _parse_csv(path)
    return _parse_topas(path)


def _float(row: dict[str, str], name: str, *, nonnegative: bool = False) -> float:
    value = row.get(name, "")
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"invalid {name}={value!r}") from error
    if not math.isfinite(parsed) or (nonnegative and parsed < 0.0):
        raise ValueError(f"invalid finite/non-negative {name}={value!r}")
    return parsed


def _int(row: dict[str, str], name: str, *, nonnegative: bool = False) -> int:
    value = row.get(name, "")
    try:
        parsed = int(value, 10)
    except ValueError as error:
        raise ValueError(f"invalid integer {name}={value!r}") from error
    if nonnegative and parsed < 0:
        raise ValueError(f"invalid non-negative {name}={value!r}")
    return parsed


def _close(lhs: float, rhs: float, *, relative: float = 1.0e-6) -> bool:
    return abs(lhs - rhs) <= max(1.0e-9, relative * max(abs(lhs), abs(rhs), 1.0))


@dataclass
class Cell:
    projectile_z: int
    projectile_a: int
    target_z: int
    target_a: int
    energy_bin_id: int
    energy_low: float
    energy_high: float
    energy: float
    weighted_track_length_mm: float = 0.0
    target_areal_density_weighted_per_mm2: float = 0.0
    density_values: list[float] = field(default_factory=list)
    collision_count: int = 0
    sum_weight: float = 0.0
    sum_weight_squared: float = 0.0
    history_count: int = 0
    censored_history_count: int = 0
    history_weight_sum: float = 0.0
    history_weight_squared_sum: float = 0.0
    censored_history_weight_sum: float = 0.0
    censored_history_weight_squared_sum: float = 0.0
    history_keys: set[tuple[int, int, int, int]] = field(default_factory=set)
    collision_keys: set[tuple[int, int, int, int, int]] = field(default_factory=set)


def _new_cell(row: dict[str, str]) -> Cell:
    projectile_z = _int(row, "projectile_z", nonnegative=True)
    projectile_a = _int(row, "projectile_a", nonnegative=True)
    target_z = _int(row, "target_z", nonnegative=True)
    target_a = _int(row, "target_a", nonnegative=True)
    energy_bin_id = _int(row, "energy_bin_id", nonnegative=True)
    low = _float(row, "energy_low_MeV_per_u", nonnegative=True)
    high = _float(row, "energy_high_MeV_per_u", nonnegative=True)
    energy = _float(row, "energy_MeV_per_u", nonnegative=True)
    if projectile_z <= 0 or projectile_a < projectile_z:
        raise ValueError(f"invalid projectile identity in row: {row}")
    if target_z <= 0 or target_a < target_z:
        raise ValueError(f"invalid target identity in row: {row}")
    if high <= low or energy < low - 1.0e-8 or energy > high + 1.0e-8:
        raise ValueError(f"energy bin does not contain its center: {row}")
    return Cell(
        projectile_z, projectile_a, target_z, target_a, energy_bin_id,
        low, high, energy
    )


def _validate_common(row: dict[str, str]) -> tuple[tuple[int, int, int, int, int], Cell]:
    key = (
        _int(row, "projectile_z", nonnegative=True),
        _int(row, "projectile_a", nonnegative=True),
        _int(row, "target_z", nonnegative=True),
        _int(row, "target_a", nonnegative=True),
        _int(row, "energy_bin_id", nonnegative=True),
    )
    return key, _new_cell(row)


def aggregate_exposure(
    input_paths: list[Path],
    output_path: Path,
    metadata_path: Path,
    *,
    provenance: dict[str, object] | None = None,
) -> dict[str, object]:
    if not input_paths:
        raise ValueError("at least one exposure input is required")

    cells: dict[tuple[int, int, int, int, int], Cell] = {}
    all_files: list[dict[str, str]] = []
    input_headers: list[dict[str, object]] = []
    for input_path in input_paths:
        rows, header, files = read_input(input_path)
        input_headers.append({"path": str(input_path), **header})
        all_files.extend(files)
        for row_number, row in enumerate(rows, 1):
            kind = row.get("record_kind", "")
            if kind not in {"exposure", "collision", "history_outcome"}:
                raise ValueError(f"{input_path}:{row_number}: unknown record_kind={kind!r}")
            key, parsed_cell = _validate_common(row)
            cell = cells.get(key)
            if cell is None:
                cell = parsed_cell
                cells[key] = cell
            elif not (
                _close(cell.energy_low, parsed_cell.energy_low)
                and _close(cell.energy_high, parsed_cell.energy_high)
                and _close(cell.energy, parsed_cell.energy)
            ):
                raise ValueError(f"inconsistent energy-bin definition for {key}")

            run_id = _int(row, "run_id")
            thread_id = _int(row, "thread_id")
            event_id = _int(row, "event_id")
            track_id = _int(row, "track_id")
            if kind == "exposure":
                length = _float(row, "track_length_mm", nonnegative=True)
                weight = _float(row, "track_weight", nonnegative=True)
                weighted_length = _float(
                    row, "weighted_track_length_mm", nonnegative=True
                )
                density = _float(
                    row, "target_number_density_per_mm3", nonnegative=True
                )
                areal = _float(
                    row,
                    "target_areal_density_weighted_per_mm2",
                    nonnegative=True,
                )
                if weight <= 0.0 or density <= 0.0:
                    raise ValueError(f"non-positive exposure weight/density: {row}")
                if not _close(weighted_length, length * weight, relative=2.0e-5):
                    raise ValueError(f"weighted track length disagrees with row fields: {row}")
                if not _close(areal, weighted_length * density, relative=2.0e-5):
                    raise ValueError(f"target areal density disagrees with row fields: {row}")
                cell.weighted_track_length_mm += weighted_length
                cell.target_areal_density_weighted_per_mm2 += areal
                cell.density_values.append(density)
            elif kind == "collision":
                count = _int(row, "collision_count", nonnegative=True)
                sequence = _int(row, "collision_sequence", nonnegative=True)
                weight = _float(row, "collision_weight_sum", nonnegative=True)
                weight_squared = _float(
                    row, "collision_weight_squared_sum", nonnegative=True
                )
                actual_energy = _float(
                    row, "collision_energy_MeV_per_u", nonnegative=True
                )
                if count <= 0 or weight <= 0.0 or weight_squared <= 0.0:
                    raise ValueError(f"collision aggregate has invalid positive moments: {row}")
                collision_moment_tolerance = 2.0e-6 * max(
                    weight * weight, count * weight_squared, 1.0
                )
                if weight * weight > (
                    count * weight_squared + collision_moment_tolerance
                ):
                    raise ValueError(f"collision aggregate violates weight-moment closure: {row}")
                if actual_energy < cell.energy_low - 1.0e-7 or actual_energy > cell.energy_high + 1.0e-7:
                    raise ValueError(f"authoritative collision energy is outside its bin: {row}")
                collision_key = (run_id, thread_id, event_id, track_id, sequence)
                if collision_key in cell.collision_keys:
                    raise ValueError(f"duplicate collision identity {collision_key}")
                cell.collision_keys.add(collision_key)
                cell.collision_count += count
                cell.sum_weight += weight
                cell.sum_weight_squared += weight_squared
            else:
                history_key = (run_id, thread_id, event_id, track_id)
                if history_key in cell.history_keys:
                    raise ValueError(
                        f"duplicate history outcome for cell {key}: {history_key}"
                    )
                cell.history_keys.add(history_key)
                count = _int(row, "history_count", nonnegative=True)
                censored = _int(row, "censored_history_count", nonnegative=True)
                history_weight = _float(row, "history_weight_sum", nonnegative=True)
                history_weight_squared = _float(
                    row, "history_weight_squared_sum", nonnegative=True
                )
                censored_weight = _float(
                    row, "censored_history_weight_sum", nonnegative=True
                )
                censored_weight_squared = _float(
                    row, "censored_history_weight_squared_sum", nonnegative=True
                )
                if count <= 0 or censored < 0 or censored > count:
                    raise ValueError(f"history outcome has invalid counts: {row}")
                moment_tolerance = 2.0e-6 * max(
                    history_weight * history_weight,
                    count * history_weight_squared,
                    1.0,
                )
                if history_weight * history_weight > (
                    count * history_weight_squared + moment_tolerance
                ):
                    raise ValueError(f"history outcome violates weight-moment closure: {row}")
                if history_weight <= 0.0 or history_weight_squared <= 0.0:
                    raise ValueError(f"history outcome has invalid weight moments: {row}")
                if censored == 0 and (censored_weight != 0.0 or censored_weight_squared != 0.0):
                    raise ValueError(f"uncensored history has nonzero censor weight: {row}")
                if censored > 0 and (
                    censored_weight <= 0.0 or censored_weight_squared <= 0.0
                ):
                    raise ValueError(f"censored history has zero censor weight: {row}")
                censor_tolerance = 2.0e-6 * max(
                    censored_weight * censored_weight,
                    censored * censored_weight_squared,
                    1.0,
                )
                if censored > 0 and censored_weight * censored_weight > (
                    censored * censored_weight_squared + censor_tolerance
                ):
                    raise ValueError(f"censored history violates weight-moment closure: {row}")
                if (
                    censored_weight > history_weight + 1.0e-6 * max(history_weight, 1.0)
                    or censored_weight_squared
                    > history_weight_squared + 1.0e-6 * max(history_weight_squared, 1.0)
                ):
                    raise ValueError(f"censored history moments exceed total moments: {row}")
                cell.history_count += count
                cell.censored_history_count += censored
                cell.history_weight_sum += history_weight
                cell.history_weight_squared_sum += history_weight_squared
                cell.censored_history_weight_sum += censored_weight
                cell.censored_history_weight_squared_sum += censored_weight_squared

    if not cells:
        raise ValueError("exposure inputs contain no rows")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RATE_INPUT_COLUMNS)
        writer.writeheader()
        for key in sorted(cells):
            cell = cells[key]
            density = ""
            if cell.density_values and all(
                _close(value, cell.density_values[0], relative=2.0e-6)
                for value in cell.density_values[1:]
            ):
                density = f"{cell.density_values[0]:.12g}"
            writer.writerow(
                {
                    "projectile_z": cell.projectile_z,
                    "projectile_a": cell.projectile_a,
                    "target_z": cell.target_z,
                    "target_a": cell.target_a,
                    "energy_bin_id": cell.energy_bin_id,
                    "energy_low_MeV_per_u": f"{cell.energy_low:.12g}",
                    "energy_high_MeV_per_u": f"{cell.energy_high:.12g}",
                    "energy_MeV_per_u": f"{cell.energy:.12g}",
                    "weighted_track_length_mm": f"{cell.weighted_track_length_mm:.12g}",
                    "target_areal_density_weighted_per_mm2": f"{cell.target_areal_density_weighted_per_mm2:.12g}",
                    "target_number_density_per_mm3": density,
                    "collision_count": cell.collision_count,
                    "sum_weight": f"{cell.sum_weight:.12g}",
                    "sum_weight_squared": f"{cell.sum_weight_squared:.12g}",
                    "history_count": cell.history_count,
                    "censored_history_count": cell.censored_history_count,
                    "history_weight_sum": f"{cell.history_weight_sum:.12g}",
                    "history_weight_squared_sum": f"{cell.history_weight_squared_sum:.12g}",
                    "censored_history_weight_sum": f"{cell.censored_history_weight_sum:.12g}",
                    "censored_history_weight_squared_sum": f"{cell.censored_history_weight_squared_sum:.12g}",
                }
            )

    samples: list[dict[str, object]] = []
    for key in sorted(cells):
        cell = cells[key]
        effective_collisions = (
            cell.sum_weight * cell.sum_weight / cell.sum_weight_squared
            if cell.sum_weight_squared > 0.0
            else 0.0
        )
        effective_histories = (
            cell.history_weight_sum * cell.history_weight_sum /
            cell.history_weight_squared_sum
            if cell.history_weight_squared_sum > 0.0
            else 0.0
        )
        samples.append(
            {
                "projectile_z": cell.projectile_z,
                "projectile_a": cell.projectile_a,
                "target_z": cell.target_z,
                "target_a": cell.target_a,
                "energy_bin_id": cell.energy_bin_id,
                "energy_low_MeV_per_u": cell.energy_low,
                "energy_high_MeV_per_u": cell.energy_high,
                "energy_MeV_per_u": cell.energy,
                "weighted_track_length_mm": cell.weighted_track_length_mm,
                "target_areal_density_weighted_per_mm2": cell.target_areal_density_weighted_per_mm2,
                "collision_count": cell.collision_count,
                "sum_weight": cell.sum_weight,
                "sum_weight_squared": cell.sum_weight_squared,
                "effective_collision_count": effective_collisions,
                "history_count": cell.history_count,
                "censored_history_count": cell.censored_history_count,
                "history_weight_sum": cell.history_weight_sum,
                "history_weight_squared_sum": cell.history_weight_squared_sum,
                "effective_history_count": effective_histories,
                "censored_history_weight_sum": cell.censored_history_weight_sum,
                "censored_history_weight_squared_sum": cell.censored_history_weight_squared_sum,
                "coverage_status": "covered" if cell.weighted_track_length_mm > 0.0 else "uncovered",
            }
        )
    metadata: dict[str, object] = {
        "format": "CINEL02_EXPOSURE_V1",
        "columns": list(EXPOSURE_COLUMNS),
        "rate_input_columns": list(RATE_INPUT_COLUMNS),
        "input_files": all_files,
        "input_headers": input_headers,
        "cell_count": len(samples),
        "samples": samples,
        "provenance": provenance or {},
        "output": {
            "path": str(output_path),
            "sha256": _sha256(output_path),
        },
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def _provenance_from_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        "campaign_uuid": args.campaign_uuid,
        "topas_version": args.topas_version,
        "geant4_version": args.geant4_version,
        "physics_list": args.physics_list,
        "hadronic_process": args.hadronic_process,
        "hadronic_model": args.hadronic_model,
        "production_cuts": args.production_cuts,
        "step_limits": args.step_limits,
        "material": args.material,
        "source": args.source,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata-output", required=True, type=Path)
    parser.add_argument("--campaign-uuid", default="")
    parser.add_argument("--topas-version", default="")
    parser.add_argument("--geant4-version", default="")
    parser.add_argument("--physics-list", default="")
    parser.add_argument("--hadronic-process", default="ionInelastic")
    parser.add_argument("--hadronic-model", default="INCLXX")
    parser.add_argument("--production-cuts", "--cuts", dest="production_cuts", default="")
    parser.add_argument("--step-limits", default="")
    parser.add_argument("--material", default="G4_WATER")
    parser.add_argument("--source", default="")
    args = parser.parse_args()
    try:
        result = aggregate_exposure(
            args.input,
            args.output,
            args.metadata_output,
            provenance=_provenance_from_args(args),
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(
        f"aggregated CINEL02 exposure: {result['cell_count']} cells, "
        f"output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
