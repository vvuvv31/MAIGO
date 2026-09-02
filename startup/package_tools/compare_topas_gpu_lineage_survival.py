#!/usr/bin/env python3
"""Compare TOPAS isotope lineage episodes with the GPU secondary ledger.

This report is descriptive only.  It keeps TOPAS generation-resolved outcomes
separate from the GPU G1 terminal ledger and never changes rates or physics.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

SPECIES = ["1H", "2H", "3H", "3He", "4He", "6He", "6Li", "7Li", "7Be", "9Be", "10Be", "8B", "10B", "11B", "10C", "11C", "12C", "6Be"]


def _gpu_rows(ledger: dict[str, Any], exposure: dict[str, Any]) -> list[dict[str, Any]]:
    species = ledger["cinel02_transition_layout"]["species"]
    n = len(species)
    counts = ledger["cinel02_queued_transition_counts"]
    kinetic = ledger["cinel02_queued_transition_kinetic_MeV"]
    reason_names = ledger["cinel02_species_terminal_reason_layout"]["reason"]
    reason_counts = ledger["cinel02_species_terminal_reason_counts"]
    exposure_rows = {row["species"]: row for row in exposure["species"]}
    rows = []
    for isotope in ("6Li", "7Li", "7Be", "9Be", "10Be"):
        index = species.index(isotope)
        birth_count = sum(int(counts[parent * n + index]) for parent in range(n))
        birth_energy = sum(float(kinetic[parent * n + index]) for parent in range(n))
        terminal = dict(
            zip(reason_names, reason_counts[index * len(reason_names):(index + 1) * len(reason_names)])
        )
        reactions = int(terminal.get("reaction_killed", 0))
        exposure_row = exposure_rows[isotope]
        rows.append(
            {
                "source": "GPU",
                "isotope": isotope,
                "generation": "G1_policy_total",
                "episode_count": birth_count,
                "reacted_count": reactions,
                "reaction_fraction": reactions / birth_count if birth_count else None,
                "path_length_mm": None,
                "birth_kinetic_energy_MeV": birth_energy,
                "tau_runtime": exposure_row["tau_runtime"],
                "tau_continuous": exposure_row.get("tau_continuous"),
                "generation_blocked_hazard": exposure_row.get("tau_blocked_continuous", exposure_row.get("tau_blocked_counterfactual")),
                "terminal_reasons": terminal,
            }
        )
    return rows


def compare(topas: dict[str, Any], ledger: dict[str, Any], exposure: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for row in topas["isotope_generation"]:
        isotope = f"{'Li' if row['atomic_number'] == 3 else 'Be'}{row['atomic_mass']}"
        if isotope not in {"Li6", "Li7", "Be7", "Be9", "Be10"}:
            continue
        item = dict(row)
        item["source"] = "TOPAS"
        item["isotope"] = isotope
        item["reaction_fraction"] = (
            row["reacted_count"] / row["episode_count"] if row["episode_count"] else None
        )
        rows.append(item)
    rows.extend(_gpu_rows(ledger, exposure))
    return {
        "schema": "TOPAS_GPU_LINEAGE_SURVIVAL_COMPARISON_V1",
        "topas_schema": topas.get("schema"),
        "gpu_exposure_schema": exposure.get("schema"),
        "rows": rows,
        "interpretation": {
            "topas_generations": "direct products of the primary reaction are G0; subsequent hadronic descendants increment generation",
            "gpu_generation": "GPU terminal ledger is the current G1-policy total; reaction candidates are limited by the configured generation gate",
            "scope": "descriptive 100k smoke; not a matched physics acceptance gate",
        },
    }


def write_outputs(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    fields = ["source", "isotope", "generation", "episode_count", "reacted_count", "reaction_fraction", "path_length_mm", "birth_kinetic_energy_MeV", "tau_runtime", "tau_continuous", "generation_blocked_hazard"]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in report["rows"]:
            writer.writerow({field: row.get(field) for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topas", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--gpu-exposure", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(
        json.loads(args.topas.read_text()),
        json.loads(args.ledger.read_text()),
        json.loads(args.gpu_exposure.read_text()),
    )
    write_outputs(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
