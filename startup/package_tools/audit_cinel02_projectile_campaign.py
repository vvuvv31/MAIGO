#!/usr/bin/env python3
"""Audit CINEL02 source projectile coverage and generated isotope products.

The campaign summary identifies which projectile/source-energy exposures were
actually run.  Optional raw files are streamed to independently count
interaction projectiles and products.  This distinguishes a missing
secondary-projectile campaign (for example 6Be) from a compiler filter that
dropped an existing campaign.  No package or rate file is modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
from audit_cinel02_unsupported import iter_records  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_summary(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"campaign summary has no header: {path}")
        required = {"projectile_z", "projectile_a", "source_energy_MeV_per_u",
                    "histories", "interactions", "products"}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"campaign summary missing fields {sorted(missing)}: {path}")
        rows = []
        for line_number, row in enumerate(reader, 2):
            try:
                parsed = {
                    "projectile_z": int(row["projectile_z"]),
                    "projectile_a": int(row["projectile_a"]),
                    "source_energy_MeV_per_u": float(row["source_energy_MeV_per_u"]),
                    "histories": int(row["histories"]),
                    "interactions": int(row["interactions"]),
                    "products": int(row["products"]),
                }
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid campaign summary row {path}:{line_number}") from error
            if parsed["projectile_z"] <= 0 or parsed["projectile_a"] < parsed["projectile_z"]:
                raise ValueError(f"invalid projectile identity at {path}:{line_number}")
            if any(parsed[name] < 0 for name in ("histories", "interactions", "products")):
                raise ValueError(f"negative campaign count at {path}:{line_number}")
            rows.append(parsed)
    return rows


def audit(summary_path: Path, raw_paths: list[Path] | None = None) -> dict[str, Any]:
    summary_rows = read_summary(summary_path)
    source_projectiles = Counter((row["projectile_z"], row["projectile_a"]) for row in summary_rows)
    source_energies: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in summary_rows:
        source_energies[(row["projectile_z"], row["projectile_a"])].append(
            row["source_energy_MeV_per_u"]
        )

    raw_interactions = 0
    raw_products = 0
    raw_projectiles: Counter[tuple[int, int]] = Counter()
    raw_targets: Counter[tuple[int, int, int, int]] = Counter()
    products: Counter[tuple[int, int]] = Counter()
    product_roles: Counter[tuple[int, int, int]] = Counter()
    product_kinetic: defaultdict[tuple[int, int], float] = defaultdict(float)
    if raw_paths:
        for path in raw_paths:
            for record, record_products in iter_records(path):
                raw_interactions += 1
                projectile = (int(record["projectile_z"]), int(record["projectile_a"]))
                target = (int(record["target_z"]), int(record["target_a"]))
                raw_projectiles[projectile] += 1
                raw_targets[(*projectile, *target)] += 1
                for product in record_products:
                    identity = (int(product["z"]), int(product["a"]))
                    role = int(product["role"])
                    raw_products += 1
                    products[identity] += 1
                    product_roles[(*identity, role)] += 1
                    product_kinetic[identity] += float(product["kinetic_energy_MeV"])

    be6 = (4, 6)
    return {
        "format": "CINEL02_PROJECTILE_CAMPAIGN_AUDIT_V1",
        "inputs": {
            "summary": {"path": str(summary_path), "sha256": _sha256(summary_path)},
            "raw": [
                {"path": str(path), "sha256": _sha256(path)} for path in (raw_paths or [])
            ],
        },
        "summary": {
            "row_count": len(summary_rows),
            "projectile_identity_count": len(source_projectiles),
            "projectile_identities": [
                {"z": z, "a": a, "campaign_count": count,
                 "source_energies_MeV_per_u": sorted(source_energies[(z, a)])}
                for (z, a), count in sorted(source_projectiles.items())
            ],
        },
        "raw": {
            "scanned": bool(raw_paths),
            "interaction_count": raw_interactions,
            "product_count": raw_products,
            "projectile_interactions": {f"Z{z}A{a}": count
                                        for (z, a), count in sorted(raw_projectiles.items())},
            "target_interactions": {f"Z{z}A{a}->Z{tz}A{ta}": count
                                    for (z, a, tz, ta), count in sorted(raw_targets.items())},
            "product_counts": {f"Z{z}A{a}": count
                               for (z, a), count in sorted(products.items())},
            "product_kinetic_energy_MeV": {f"Z{z}A{a}": value
                                            for (z, a), value in sorted(product_kinetic.items())},
            "product_roles": {f"Z{z}A{a}:role{role}": count
                              for (z, a, role), count in sorted(product_roles.items())},
        },
        "be6_assessment": {
            "identity": {"z": 4, "a": 6},
            "source_campaign_present": be6 in source_projectiles,
            "raw_projectile_interactions": raw_projectiles.get(be6, 0),
            "raw_direct_product_count": product_roles.get((4, 6, 0), 0),
            "raw_product_count_all_roles": products.get(be6, 0),
            "conclusion": (
                "secondary-projectile campaign absent; Be-6 is generated as a direct child "
                "but has no independent exposure/rate/package support"
                if be6 not in source_projectiles and products.get(be6, 0) > 0
                else "coverage requires manual review"
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--raw", nargs="*", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = audit(args.summary, args.raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["be6_assessment"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
