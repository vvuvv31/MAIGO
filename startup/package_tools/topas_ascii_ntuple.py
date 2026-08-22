#!/usr/bin/env python3
"""Strict reader for EnergyLossFluctuationNtuple TOPAS ASCII output."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Callable, TypeVar


ENERGY_LOSS_COLUMNS = (
    "Run ID",
    "Event ID",
    "Thread ID",
    "Primary Track ID",
    "Atomic Number Z",
    "Mass Number A",
    "Material Name",
    "Entry Kinetic Energy (MeV)",
    "Exit Kinetic Energy (MeV)",
    "Primary Kinetic Energy Loss (MeV)",
    "Primary Local Deposit (MeV)",
    "Primary Path Length (mm)",
    "Primary Step Count",
    "Material Consistent",
    "Completed",
    "Completion Status",
)

_COUNT_PATTERNS = {
    "original_histories": re.compile(
        r"^\s*Number of Original Histories:\s*(\d+)\s*$"
    ),
    "scored_entries": re.compile(
        r"^\s*Number of Scored Entries:\s*(\d+)\s*$"
    ),
}
_COLUMN_PATTERN = re.compile(r"^\s*(\d+):\s*(.*?)\s*$")


@dataclass(frozen=True)
class TopasNtupleHeader:
    original_histories: int
    scored_entries: int
    columns: tuple[str, ...]


@dataclass(frozen=True)
class EnergyLossFluctuationRow:
    run_id: int
    event_id: int
    thread_id: int
    primary_track_id: int
    atomic_number: int
    mass_number: int
    material_name: str
    entry_energy_mev: float
    exit_energy_mev: float
    kinetic_energy_loss_mev: float
    primary_local_deposit_mev: float
    path_length_mm: float
    step_count: int
    material_consistent: bool
    completed: bool
    completion_status: str


def _resolve_paths(path: Path) -> tuple[Path, Path]:
    if path.suffix == ".header":
        return path, path.with_suffix(".phsp")
    if path.suffix == ".phsp":
        return path.with_suffix(".header"), path
    return Path(f"{path}.header"), Path(f"{path}.phsp")


def _read_header(path: Path) -> TopasNtupleHeader:
    if not path.is_file():
        raise FileNotFoundError(f"TOPAS ntuple header is missing: {path}")

    counts: dict[str, int] = {}
    indexed_columns: list[tuple[int, str]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        for key, pattern in _COUNT_PATTERNS.items():
            match = pattern.match(line)
            if match:
                if key in counts:
                    raise ValueError(f"{path}:{line_number}: duplicate {key}")
                counts[key] = int(match.group(1))
                break
        else:
            match = _COLUMN_PATTERN.match(line)
            if match:
                name = match.group(2)
                if not name:
                    raise ValueError(f"{path}:{line_number}: empty column name")
                indexed_columns.append((int(match.group(1)), name))

    missing = [key for key in _COUNT_PATTERNS if key not in counts]
    if missing:
        raise ValueError(f"{path}: missing header fields: {', '.join(missing)}")
    expected_indices = list(range(1, len(indexed_columns) + 1))
    actual_indices = [index for index, _ in indexed_columns]
    if actual_indices != expected_indices:
        raise ValueError(
            f"{path}: column indices must be contiguous from 1; got {actual_indices}"
        )
    columns = tuple(name for _, name in indexed_columns)
    if columns != ENERGY_LOSS_COLUMNS:
        raise ValueError(
            f"{path}: EnergyLossFluctuationNtuple schema mismatch; "
            f"expected {ENERGY_LOSS_COLUMNS}, got {columns}"
        )
    return TopasNtupleHeader(
        original_histories=counts["original_histories"],
        scored_entries=counts["scored_entries"],
        columns=columns,
    )


def _parse_int(value: str, location: str) -> int:
    try:
        # Reject values such as 1.0 that int(float(...)) would silently accept.
        return int(value, 10)
    except ValueError as error:
        raise ValueError(f"{location}: expected integer, got {value!r}") from error


def _parse_float(value: str, location: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"{location}: expected floating-point value, got {value!r}") from error
    if not math.isfinite(parsed):
        raise ValueError(f"{location}: non-finite floating-point value {value!r}")
    return parsed


def _parse_bool(value: str, location: str) -> bool:
    normalized = value.lower()
    if normalized in {"1", "true"}:
        return True
    if normalized in {"0", "false"}:
        return False
    raise ValueError(f"{location}: expected boolean 0/1 or true/false, got {value!r}")


T = TypeVar("T")


def _field(
    values: list[str], index: int, parser: Callable[[str, str], T],
    path: Path, line_number: int
) -> T:
    return parser(
        values[index],
        f"{path}:{line_number}, column {index + 1} ({ENERGY_LOSS_COLUMNS[index]})",
    )


def _parse_row(path: Path, line_number: int, values: list[str]) -> EnergyLossFluctuationRow:
    if len(values) != len(ENERGY_LOSS_COLUMNS):
        raise ValueError(
            f"{path}:{line_number}: expected {len(ENERGY_LOSS_COLUMNS)} fields, "
            f"got {len(values)}"
        )
    return EnergyLossFluctuationRow(
        run_id=_field(values, 0, _parse_int, path, line_number),
        event_id=_field(values, 1, _parse_int, path, line_number),
        thread_id=_field(values, 2, _parse_int, path, line_number),
        primary_track_id=_field(values, 3, _parse_int, path, line_number),
        atomic_number=_field(values, 4, _parse_int, path, line_number),
        mass_number=_field(values, 5, _parse_int, path, line_number),
        material_name=values[6],
        entry_energy_mev=_field(values, 7, _parse_float, path, line_number),
        exit_energy_mev=_field(values, 8, _parse_float, path, line_number),
        kinetic_energy_loss_mev=_field(values, 9, _parse_float, path, line_number),
        primary_local_deposit_mev=_field(values, 10, _parse_float, path, line_number),
        path_length_mm=_field(values, 11, _parse_float, path, line_number),
        step_count=_field(values, 12, _parse_int, path, line_number),
        material_consistent=_field(values, 13, _parse_bool, path, line_number),
        completed=_field(values, 14, _parse_bool, path, line_number),
        completion_status=values[15],
    )


def read_energy_loss_fluctuation_ntuple(
    path: str | Path,
) -> tuple[TopasNtupleHeader, list[EnergyLossFluctuationRow]]:
    """Read a TOPAS output stem, .header, or .phsp and enforce its schema."""

    header_path, phsp_path = _resolve_paths(Path(path))
    header = _read_header(header_path)
    if not phsp_path.is_file():
        raise FileNotFoundError(f"TOPAS ntuple phase space is missing: {phsp_path}")

    rows: list[EnergyLossFluctuationRow] = []
    with phsp_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            values = line.split()
            if not values:
                continue
            rows.append(_parse_row(phsp_path, line_number, values))
    if len(rows) != header.scored_entries:
        raise ValueError(
            f"{phsp_path}: header declares {header.scored_entries} scored entries, "
            f"but parsed {len(rows)}"
        )
    return header, rows
