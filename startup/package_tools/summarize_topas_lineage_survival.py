#!/usr/bin/env python3
"""Summarize TOPAS CarbonLineageSurvivalNtuple episode output.

The scorer emits one row per Li/Be transport episode.  This command keeps the
raw episode semantics intact while producing isotope-by-generation aggregates
for comparison with the GPU optical-depth ledger.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

_HEADER_COUNT = {
    "Number of Original Histories": "original_histories",
    "Number of Scored Entries": "scored_entries",
}
_COLUMN_RE = re.compile(r"^\s*(\d+):\s*(.*?)\s*$")
_COLUMN_NAMES = {
    "Record Kind": "record_kind",
    "Run ID": "run_id",
    "Thread ID": "thread_id",
    "Event ID": "event_id",
    "Track ID": "track_id",
    "Parent Track ID": "parent_id",
    "Atomic Number Z": "atomic_number",
    "Atomic Mass A": "atomic_mass",
    "Transport Generation": "generation",
    "Episode Index": "episode_index",
    "Step Count": "step_count",
    "Hadronic Interaction": "reacted",
    "Censored At Event End": "censored",
    "Weight": "weight",
    "Birth Kinetic Energy (MeV)": "birth_kinetic_energy_MeV",
    "Terminal Kinetic Energy (MeV)": "terminal_kinetic_energy_MeV",
    "Interaction Kinetic Energy (MeV)": "interaction_kinetic_energy_MeV",
    "Path Length (mm)": "path_length_mm",
    "Birth X (mm)": "birth_x_mm",
    "Birth Y (mm)": "birth_y_mm",
    "Birth Z (mm)": "birth_z_mm",
    "Terminal X (mm)": "terminal_x_mm",
    "Terminal Y (mm)": "terminal_y_mm",
    "Terminal Z (mm)": "terminal_z_mm",
    "Creator Process": "creator_process",
    "Terminal Reason": "terminal_reason",
    "Terminal Process": "terminal_process",
}
_REQUIRED = tuple(_COLUMN_NAMES.values())
_TRACKED = {(3, 6), (3, 7), (4, 7), (4, 9), (4, 10)}


def read_header(path: Path) -> tuple[dict[str, int], list[str]]:
    counts: dict[str, int] = {}
    columns: dict[int, str] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            stripped = line.strip()
            for label, key in _HEADER_COUNT.items():
                prefix = label + ":"
                if stripped.startswith(prefix):
                    counts[key] = int(stripped[len(prefix):].strip())
            match = _COLUMN_RE.match(line)
            if match:
                index = int(match.group(1))
                label = match.group(2)
                if label.endswith("]") and " [" in label:
                    label = label.rsplit(" [", 1)[0]
                if label not in _COLUMN_NAMES:
                    raise ValueError(f"{path}: unsupported column {label!r}")
                columns[index] = _COLUMN_NAMES[label]
    if not columns:
        raise ValueError(f"{path}: no n-tuple columns found")
    ordered = [columns[index] for index in sorted(columns)]
    missing = [name for name in _REQUIRED if name not in ordered]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return counts, ordered


def resolve_paths(path: Path) -> tuple[Path, Path]:
    if path.suffix == ".header":
        return path, path.with_suffix(".phsp")
    if path.suffix == ".phsp":
        return path.with_suffix(".header"), path
    return Path(str(path) + ".header"), Path(str(path) + ".phsp")


def read_rows(path: Path, columns: list[str]) -> Iterable[dict[str, str]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            values = line.split()
            if len(values) != len(columns):
                raise ValueError(
                    f"{path}:{line_number}: expected {len(columns)} fields, got {len(values)}"
                )
            yield dict(zip(columns, values))


def _float(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"non-finite {key}: {value}")
    return value


def summarize(input_path: Path) -> dict:
    header_path, data_path = resolve_paths(input_path)
    header_counts, columns = read_header(header_path)
    aggregates: dict[tuple[int, int, int], dict[str, float | int]] = defaultdict(
        lambda: {
            "episode_count": 0,
            "reacted_count": 0,
            "censored_count": 0,
            "path_length_mm": 0.0,
            "birth_kinetic_energy_MeV": 0.0,
            "interaction_kinetic_energy_MeV": 0.0,
            "terminal_kinetic_energy_MeV": 0.0,
            "weight_sum": 0.0,
        }
    )
    reasons: dict[tuple[int, int, int], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    seen: set[tuple[int, int, int, int, int]] = set()
    row_count = 0
    unexpected: list[dict[str, int]] = []

    for row in read_rows(data_path, columns):
        row_count += 1
        z = int(row["atomic_number"])
        a = int(row["atomic_mass"])
        key = (z, a)
        if key not in _TRACKED:
            unexpected.append({"z": z, "a": a})
            continue
        generation = int(row["generation"])
        identity = (
            int(row["run_id"]),
            int(row["thread_id"]),
            int(row["event_id"]),
            int(row["track_id"]),
            int(row["episode_index"]),
        )
        if identity in seen:
            raise ValueError(f"duplicate episode identity {identity}")
        seen.add(identity)
        item = aggregates[(z, a, generation)]
        item["episode_count"] += 1
        reacted = int(row["reacted"])
        censored = int(row["censored"])
        if reacted not in (0, 1) or censored not in (0, 1):
            raise ValueError(f"invalid flags in episode {identity}")
        item["reacted_count"] += reacted
        item["censored_count"] += censored
        item["path_length_mm"] += _float(row, "path_length_mm")
        item["birth_kinetic_energy_MeV"] += _float(row, "birth_kinetic_energy_MeV")
        item["interaction_kinetic_energy_MeV"] += _float(row, "interaction_kinetic_energy_MeV")
        item["terminal_kinetic_energy_MeV"] += _float(row, "terminal_kinetic_energy_MeV")
        item["weight_sum"] += _float(row, "weight")
        reason = row["terminal_reason"]
        reasons[(z, a, generation)][reason] += 1

    if header_counts.get("scored_entries") is not None and row_count != header_counts["scored_entries"]:
        raise ValueError(
            f"{data_path}: header scored entries {header_counts['scored_entries']} != rows {row_count}"
        )
    if unexpected:
        raise ValueError(f"unexpected isotope rows, first={unexpected[:3]}")

    rows = []
    for (z, a, generation), item in sorted(aggregates.items()):
        reason_counts = dict(sorted(reasons[(z, a, generation)].items()))
        if sum(reason_counts.values()) != item["episode_count"]:
            raise ValueError("terminal reason partition does not close")
        if item["reacted_count"] != reason_counts.get("hadronic_interaction", 0):
            raise ValueError("reacted flag does not match hadronic interaction reason")
        if item["censored_count"] != reason_counts.get("event_end_censored", 0):
            raise ValueError("censored flag does not match event-end reason")
        rows.append(
            {
                "atomic_number": z,
                "atomic_mass": a,
                "generation": generation,
                **item,
                "terminal_reasons": reason_counts,
            }
        )

    isotope_totals: dict[str, dict[str, float | int]] = {}
    for row in rows:
        name = f"Z{row['atomic_number']}A{row['atomic_mass']}"
        total = isotope_totals.setdefault(
            name,
            {
                "episode_count": 0,
                "reacted_count": 0,
                "censored_count": 0,
                "path_length_mm": 0.0,
                "birth_kinetic_energy_MeV": 0.0,
                "interaction_kinetic_energy_MeV": 0.0,
                "terminal_kinetic_energy_MeV": 0.0,
                "weight_sum": 0.0,
            },
        )
        for key in (
            "episode_count",
            "reacted_count",
            "censored_count",
            "path_length_mm",
            "birth_kinetic_energy_MeV",
            "interaction_kinetic_energy_MeV",
            "terminal_kinetic_energy_MeV",
            "weight_sum",
        ):
            total[key] += row[key]

    return {
        "schema": "TOPAS_CARBON_LINEAGE_SURVIVAL_V1",
        "input_header": str(header_path),
        "input_data": str(data_path),
        "header": header_counts,
        "row_count": row_count,
        "episode_count": sum(int(row["episode_count"]) for row in rows),
        "tracked_isotopes": ["Li6", "Li7", "Be7", "Be9", "Be10"],
        "isotope_generation": rows,
        "isotope_totals": isotope_totals,
    }


def write_outputs(report: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    fields = [
        "atomic_number",
        "atomic_mass",
        "generation",
        "episode_count",
        "reacted_count",
        "censored_count",
        "path_length_mm",
        "birth_kinetic_energy_MeV",
        "interaction_kinetic_energy_MeV",
        "terminal_kinetic_energy_MeV",
        "weight_sum",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in report["isotope_generation"]:
            writer.writerow({field: row[field] for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="lineage_survival stem, .header or .phsp")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.input)
    write_outputs(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
