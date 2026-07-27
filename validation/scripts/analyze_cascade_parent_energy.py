#!/usr/bin/env python3
"""Parent-MeV/u conditioned light-isotope spectra from TOPAS cascade packages.

Also reweights package product yields to a GPU parent-energy histogram so
gen1 comparisons are not dominated by different parent-energy mixtures.

Example:
  python3 validation/scripts/analyze_cascade_parent_energy.py \\
    --products validation/results/topas_400MeVu_cascade_g4_11_3_2_100k_products.csv.gz \\
    --interactions validation/results/topas_400MeVu_cascade_g4_11_3_2_100k_interactions.csv.gz \\
    --gpu-parent-mevu out/birth_spectrum/gpu_400_parent_mevu.csv \\
    --output-json validation/results/birth_spectrum_energy_suite/cascade_parent_energy_400.json
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

SPECIES_ZA = {
    (1, 1): "proton",
    (1, 2): "deuteron",
    (1, 3): "triton",
    (2, 3): "he3",
    (2, 4): "he4",
}
PARENT_BIN_WIDTH = 10.0  # MeV/u


def open_rows(path: Path) -> Iterable[Dict[str, str]]:
    if str(path).endswith(".gz"):
        fh = gzip.open(path, "rt", encoding="utf-8", newline="")
    else:
        fh = open(path, "r", encoding="utf-8", newline="")
    with fh:
        yield from csv.DictReader(fh)


def load_gpu_parent_hist(path: Optional[Path]) -> Dict[str, Dict[int, float]]:
    """species -> parent_mevu_bin -> count (from GPU birth _parent_mevu.csv)."""
    if path is None or not path.exists():
        return {}
    out: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            low = float(row["parent_mevu_bin_low"])
            b = int(round(low / PARENT_BIN_WIDTH))
            out[row["species"]][b] += float(row["count"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--products", type=Path, required=True)
    parser.add_argument("--interactions", type=Path, required=True)
    parser.add_argument("--gpu-parent-mevu", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--histories", type=int, default=100000)
    args = parser.parse_args()

    parent: Dict[int, Tuple[int, int, float]] = {}
    for row in open_rows(args.interactions):
        iid = int(row["interaction_id"])
        parent[iid] = (
            int(float(row.get("projectile_Z", row.get("projectile_z", 0)))),
            int(float(row.get("projectile_A", row.get("projectile_a", 0)))),
            float(
                row.get(
                    "incident_energy_MeV_per_u",
                    row.get("incident_energy_mev_per_u", 0.0),
                )
            ),
        )

    # species -> parent_bin -> {count, ke_sum, mevu_sum}
    stats: Dict[str, Dict[int, Dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: {"count": 0.0, "ke_sum": 0.0, "mevu_sum": 0.0})
    )
    parent_type_counts: Dict[str, Dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )

    for row in open_rows(args.products):
        z = int(float(row.get("Z", row.get("atomic_number_Z", 0))))
        a = int(float(row.get("A", row.get("mass_number_A", 0))))
        sp = SPECIES_ZA.get((z, a))
        if sp is None:
            continue
        iid = int(row["interaction_id"])
        if iid not in parent:
            continue
        pz, pa, pe = parent[iid]
        pbin = int(pe / PARENT_BIN_WIDTH)
        ke = float(row["kinetic_energy_MeV"])
        mevu = ke / float(a) if a > 0 else 0.0
        s = stats[sp][pbin]
        s["count"] += 1.0
        s["ke_sum"] += ke
        s["mevu_sum"] += mevu
        ptype = "C12" if (pz, pa) == (6, 12) else "fragment"
        parent_type_counts[sp][ptype] += 1

    gpu_hist = load_gpu_parent_hist(args.gpu_parent_mevu)

    species_report = {}
    for sp, bins in stats.items():
        total_c = sum(v["count"] for v in bins.values())
        total_ke = sum(v["ke_sum"] for v in bins.values())
        total_mevu = sum(v["mevu_sum"] for v in bins.values())
        record = {
            "count": total_c,
            "yield_per_primary": total_c / args.histories if args.histories else 0.0,
            "mean_ke_MeV": total_ke / total_c if total_c else 0.0,
            "mean_mevu": total_mevu / total_c if total_c else 0.0,
            "by_parent_mevu_bin": {
                f"{b * int(PARENT_BIN_WIDTH)}-{(b + 1) * int(PARENT_BIN_WIDTH)}": {
                    "count": v["count"],
                    "mean_ke_MeV": v["ke_sum"] / v["count"] if v["count"] else 0.0,
                    "mean_mevu": v["mevu_sum"] / v["count"] if v["count"] else 0.0,
                }
                for b, v in sorted(bins.items())
                if v["count"] >= 20
            },
            "parent_type_counts": dict(parent_type_counts[sp]),
        }

        gbins = gpu_hist.get(sp, {})
        gsum = sum(gbins.values())
        if gsum > 0:
            w_coverage = 0.0
            w_ke = 0.0
            w_mevu = 0.0
            for b, gcount in gbins.items():
                if b not in bins or bins[b]["count"] <= 0:
                    continue
                w = gcount / gsum
                mean_ke = bins[b]["ke_sum"] / bins[b]["count"]
                mean_mevu = bins[b]["mevu_sum"] / bins[b]["count"]
                w_coverage += w
                w_ke += w * mean_ke
                w_mevu += w * mean_mevu
            record["reweighted"] = {
                "coverage_parent_bins": w_coverage,
                "mean_ke_MeV_reweighted_to_gpu_parent": w_ke if w_coverage else None,
                "mean_mevu_reweighted_to_gpu_parent": w_mevu if w_coverage else None,
                "gpu_parent_hist_sum": gsum,
            }
        else:
            record["reweighted"] = {"note": "no GPU parent histogram for species"}
        species_report[sp] = record

    report = {
        "products": str(args.products),
        "interactions": str(args.interactions),
        "gpu_parent_mevu": str(args.gpu_parent_mevu) if args.gpu_parent_mevu else None,
        "parent_bin_width_MeVu": PARENT_BIN_WIDTH,
        "species": species_report,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {args.output_json}")
    for sp, rec in species_report.items():
        rw = rec.get("reweighted", {})
        print(
            f"{sp:8s} package_mean_ke={rec['mean_ke_MeV']:.1f}  "
            f"reweight_mean_ke={rw.get('mean_ke_MeV_reweighted_to_gpu_parent')}"
        )


if __name__ == "__main__":
    main()
