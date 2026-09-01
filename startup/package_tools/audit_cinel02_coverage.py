#!/usr/bin/env python3
"""Stream CINEL02 raw events and build a statistical coverage qualification."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cinel02  # noqa: E402
from audit_cinel02_unsupported import iter_records  # noqa: E402


SPECIAL_MINIMUM = {(6, 12): 1000, (1, 1): 1000, (2, 4): 1000}
DEFAULT_MINIMUM = 500


@dataclass
class Moments:
    n: int = 0
    total: float = 0.0
    total2: float = 0.0

    def add(self, value: float) -> None:
        self.n += 1
        self.total += value
        self.total2 += value * value

    def mean(self) -> float:
        return self.total / self.n if self.n else 0.0

    def ci95(self) -> list[float]:
        if self.n < 2:
            return [self.mean(), self.mean()]
        variance = max(0.0, (self.total2 - self.total * self.total / self.n) / (self.n - 1))
        half = 1.96 * math.sqrt(variance / self.n)
        return [self.mean() - half, self.mean() + half]


@dataclass
class Cell:
    events: int = 0
    sum_w: float = 0.0
    sum_w2: float = 0.0
    products: int = 0
    multiplicity: Moments = field(default_factory=Moments)
    birth_e_per_u: Moments = field(default_factory=Moments)
    parent_survival: Moments = field(default_factory=Moments)
    local_energy: Moments = field(default_factory=Moments)
    neutral_energy: Moments = field(default_factory=Moments)
    unsupported_energy: Moments = field(default_factory=Moments)
    yields: dict[tuple[int, int], int] = field(default_factory=lambda: defaultdict(int))
    species_birth_e_per_u: dict[tuple[int, int], Moments] = field(
        default_factory=lambda: defaultdict(Moments))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(raw_paths: list[Path], *, campaign_uuid: str, energy_min: float = 0.0,
          energy_width: float = 1.0, raw_hashes: dict[str, str] | None = None) -> dict[str, Any]:
    cells: dict[tuple[int, int, int, int, int], Cell] = defaultdict(Cell)
    identities: dict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)
    for path in raw_paths:
        for record, products in iter_records(path):
            energy = float(record["collision_energy_MeV_per_u"])
            ebin = math.floor((energy - energy_min) / energy_width)
            if ebin < 0:
                raise ValueError(f"collision energy below grid: {energy}")
            projectile = (int(record["projectile_z"]), int(record["projectile_a"]))
            target = (int(record["target_z"]), int(record["target_a"]))
            key = (*projectile, *target, ebin)
            identities[projectile].add(target)
            cell = cells[key]
            weight = float(record["track_weight"])
            cell.events += 1
            cell.sum_w += weight
            cell.sum_w2 += weight * weight
            cell.products += len(products)
            cell.multiplicity.add(float(len(products)))
            cell.local_energy.add(float(record["process_local_deposit_MeV"]))
            cell.unsupported_energy.add(float(record["unsupported_product_energy_MeV"]))
            parent = 0.0
            neutral = 0.0
            for raw_product in products:
                product = cinel02._normalise_product(raw_product)
                role = int(product["role"])
                kinetic = float(product["kinetic_energy_MeV"])
                if role == cinel02.PRODUCT_PARENT_CONTINUATION:
                    parent = 1.0
                if int(product["z"]) <= 0:
                    neutral += kinetic
                if role == cinel02.PRODUCT_DIRECT_SECONDARY and int(product["z"]) > 0:
                    species = (int(product["z"]), int(product["a"]))
                    cell.yields[species] += 1
                    if species[1] > 0:
                        cell.birth_e_per_u.add(kinetic / species[1])
                        cell.species_birth_e_per_u[species].add(kinetic / species[1])
            cell.parent_survival.add(parent)
            cell.neutral_energy.add(neutral)

    rows: list[dict[str, Any]] = []
    for key, cell in sorted(cells.items()):
        zp, ap, zt, at, ebin = key
        minimum = SPECIAL_MINIMUM.get((zp, ap), DEFAULT_MINIMUM)
        effective = cell.sum_w * cell.sum_w / cell.sum_w2 if cell.sum_w2 else 0.0
        gates = {"minimum_event_count": cell.events >= minimum,
                 "minimum_effective_event_count": effective >= minimum}
        rows.append({
            "projectile_z": zp, "projectile_a": ap, "target_z": zt, "target_a": at,
            "energy_bin_id": ebin,
            "energy_low_MeV_per_u": energy_min + ebin * energy_width,
            "energy_high_MeV_per_u": energy_min + (ebin + 1) * energy_width,
            "raw_event_count": cell.events, "effective_event_count": effective,
            "product_count": cell.products, "rate": None,
            "mean_multiplicity": cell.multiplicity.mean(),
            "mean_multiplicity_ci95": cell.multiplicity.ci95(),
            "species_yield_per_event": {f"Z{z}A{a}": count / cell.events for (z, a), count in sorted(cell.yields.items())},
            "species_mean_birth_energy_MeV_per_u": {
                f"Z{z}A{a}": moments.mean()
                for (z, a), moments in sorted(cell.species_birth_e_per_u.items())
            },
            "mean_birth_energy_MeV_per_u": cell.birth_e_per_u.mean(),
            "mean_birth_energy_MeV_per_u_ci95": cell.birth_e_per_u.ci95(),
            "parent_survival_fraction": cell.parent_survival.mean(),
            "mean_local_energy_MeV": cell.local_energy.mean(),
            "mean_neutral_energy_MeV": cell.neutral_energy.mean(),
            "mean_unsupported_energy_MeV": cell.unsupported_energy.mean(),
            "maximum_uniform_event_reuse": effective,
            "gates": gates, "qualified": all(gates.values()),
        })

    gaps: list[dict[str, Any]] = []
    for projectile, targets in sorted(identities.items()):
        for required_target in ((1, 1), (8, 16)):
            if required_target not in targets:
                gaps.append({"kind": "missing_target", "projectile_z": projectile[0],
                             "projectile_a": projectile[1], "target_z": required_target[0],
                             "target_a": required_target[1]})
        for target in sorted(targets):
            bins = sorted(k[4] for k in cells if k[:4] == (*projectile, *target))
            occupied = set(bins)
            for ebin in range(bins[0], bins[-1] + 1):
                if ebin not in occupied:
                    gaps.append({"kind": "energy_gap", "projectile_z": projectile[0],
                                 "projectile_a": projectile[1], "target_z": target[0],
                                 "target_a": target[1], "energy_bin_id": ebin})

    files = [{"path": str(path), "sha256": (raw_hashes or {}).get(str(path)) or _sha256(path)}
             for path in raw_paths]
    qualified = bool(rows) and all(row["qualified"] for row in rows) and not gaps
    return {
        "format": cinel02.QUALIFICATION_SCHEMA, "campaign_uuid": campaign_uuid,
        "structurally_valid": True, "qualified": qualified,
        "energy_grid": {"minimum_MeV_per_u": energy_min, "width_MeV_per_u": energy_width},
        "thresholds": {"C12_p_He4_minimum_events": 1000, "other_isotope_minimum_events": 500},
        "raw_files": files, "cells": rows, "energy_gaps": gaps,
        "summary": {"cell_count": len(rows), "qualified_cell_count": sum(r["qualified"] for r in rows),
                    "unqualified_cell_count": sum(not r["qualified"] for r in rows),
                    "gap_count": len(gaps), "interaction_count": sum(r["raw_event_count"] for r in rows)},
    }


def write_outputs(report: dict[str, Any], output: Path, csv_output: Path | None) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if csv_output:
        csv_output.parent.mkdir(parents=True, exist_ok=True)
        fields = ("projectile_z", "projectile_a", "target_z", "target_a", "energy_bin_id",
                  "raw_event_count", "effective_event_count", "mean_multiplicity",
                  "parent_survival_fraction", "mean_local_energy_MeV", "mean_neutral_energy_MeV",
                  "mean_unsupported_energy_MeV", "qualified")
        with csv_output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows({name: row[name] for name in fields} for row in report["cells"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar", type=Path)
    parser.add_argument("--raw", nargs="+", type=Path)
    parser.add_argument("--campaign-uuid")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--csv-output", type=Path)
    args = parser.parse_args()
    metadata: dict[str, Any] = {}
    if args.sidecar:
        metadata = json.loads(args.sidecar.read_text(encoding="utf-8"))
    paths = args.raw or [Path(entry["path"]) for entry in metadata.get("raw_files", [])]
    campaign = args.campaign_uuid or metadata.get("campaign_uuid")
    if not paths or not campaign:
        parser.error("raw paths and campaign UUID must be provided directly or by --sidecar")
    hashes = {entry["path"]: entry["sha256"] for entry in metadata.get("raw_files", [])}
    report = audit(paths, campaign_uuid=campaign, raw_hashes=hashes)
    write_outputs(report, args.output, args.csv_output)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
