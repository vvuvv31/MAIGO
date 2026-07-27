#!/usr/bin/env python3
"""Compare GPU joint parent×product MeV/u histograms to TOPAS cascade package.

Reads GPU `_parent_product_mevu.csv` (generation-split) and builds the same joint
histogram from TOPAS cascade products+interactions. Reports per parent-energy
bin the product mean MeV/u ratio for a chosen species and generation.

Example:
  python3 validation/scripts/compare_birth_joint_parent_product.py \\
    --gpu-joint out/birth_spectrum/gpu_400_parent_product_mevu.csv \\
    --products validation/results/topas_400MeVu_cascade_g4_11_3_2_100k_products.csv.gz \\
    --interactions validation/results/topas_400MeVu_cascade_g4_11_3_2_100k_interactions.csv.gz \\
    --species he4 --generation 1 \\
    --output-json validation/results/birth_spectrum_energy_suite/he4_gen1_joint.json
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

PARENT_WIDTH = 10.0
PRODUCT_WIDTH = 2.0
SPECIES_ZA = {
    "proton": (1, 1),
    "deuteron": (1, 2),
    "triton": (1, 3),
    "he3": (2, 3),
    "he4": (2, 4),
}


def open_rows(path: Path) -> Iterable[Dict[str, str]]:
    if str(path).endswith(".gz"):
        fh = gzip.open(path, "rt", encoding="utf-8", newline="")
    else:
        fh = open(path, "r", encoding="utf-8", newline="")
    with fh:
        yield from csv.DictReader(fh)


def load_gpu_joint(
    path: Path, species: str, generation: int
) -> Dict[int, Dict[int, float]]:
    """parent_bin -> product_bin -> count"""
    out: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["species"] != species:
                continue
            if int(row["generation"]) != generation:
                continue
            pbin = int(round(float(row["parent_mevu_bin_low"]) / PARENT_WIDTH))
            ebin = int(round(float(row["product_mevu_bin_low"]) / PRODUCT_WIDTH))
            out[pbin][ebin] += float(row["count"])
    return out


def load_topas_joint(
    products: Path, interactions: Path, species: str
) -> Dict[int, Dict[int, float]]:
    za = SPECIES_ZA[species]
    parent: Dict[int, Tuple[int, int, float]] = {}
    for row in open_rows(interactions):
        iid = int(row["interaction_id"])
        pz = int(float(row.get("projectile_Z", row.get("projectile_z", 0))))
        pa = int(float(row.get("projectile_A", row.get("projectile_a", 0))))
        pe = float(
            row.get(
                "incident_energy_MeV_per_u",
                row.get("incident_energy_mev_per_u", 0.0),
            )
        )
        parent[iid] = (pz, pa, pe)

    out: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for row in open_rows(products):
        z = int(float(row.get("Z", 0)))
        a = int(float(row.get("A", 0)))
        if (z, a) != za:
            continue
        iid = int(row["interaction_id"])
        if iid not in parent:
            continue
        _, _, pe = parent[iid]
        ke = float(row["kinetic_energy_MeV"])
        mevu = ke / float(a) if a > 0 else 0.0
        pbin = int(pe / PARENT_WIDTH)
        ebin = int(mevu / PRODUCT_WIDTH)
        out[pbin][ebin] += 1.0
    return out


def mean_product_mevu(bins: Dict[int, float]) -> float:
    num = 0.0
    den = 0.0
    for ebin, c in bins.items():
        center = (ebin + 0.5) * PRODUCT_WIDTH
        num += c * center
        den += c
    return num / den if den > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-joint", type=Path, required=True)
    parser.add_argument("--products", type=Path, required=True)
    parser.add_argument("--interactions", type=Path, required=True)
    parser.add_argument("--species", default="he4")
    parser.add_argument("--generation", type=int, default=1)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--min-count", type=int, default=30)
    args = parser.parse_args()

    gpu = load_gpu_joint(args.gpu_joint, args.species, args.generation)
    top = load_topas_joint(args.products, args.interactions, args.species)

    rows = []
    for pbin in sorted(set(gpu.keys()) | set(top.keys())):
        g = gpu.get(pbin, {})
        t = top.get(pbin, {})
        gc = sum(g.values())
        tc = sum(t.values())
        if gc < args.min_count and tc < args.min_count:
            continue
        gm = mean_product_mevu(g)
        tm = mean_product_mevu(t)
        rows.append(
            {
                "parent_mevu_low": pbin * PARENT_WIDTH,
                "parent_mevu_high": (pbin + 1) * PARENT_WIDTH,
                "gpu_count": gc,
                "topas_count": tc,
                "gpu_mean_product_mevu": gm,
                "topas_mean_product_mevu": tm,
                "mean_product_mevu_ratio": (gm / tm) if tm > 0 else None,
            }
        )

    # Overall counts
    report = {
        "species": args.species,
        "generation": args.generation,
        "gpu_joint": str(args.gpu_joint),
        "by_parent_mevu_bin": rows,
        "summary": {
            "gpu_total_count": sum(r["gpu_count"] for r in rows),
            "topas_total_count": sum(r["topas_count"] for r in rows),
            "mean_ratio_unweighted": (
                sum(
                    r["mean_product_mevu_ratio"]
                    for r in rows
                    if r["mean_product_mevu_ratio"] is not None
                )
                / max(
                    1,
                    sum(
                        1
                        for r in rows
                        if r["mean_product_mevu_ratio"] is not None
                    ),
                )
            ),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {args.output_json}")
    print("parent_MeVu   gpu_n  topas_n  gpu_mean  topas_mean  ratio")
    for r in rows:
        print(
            f"{r['parent_mevu_low']:3.0f}-{r['parent_mevu_high']:3.0f}  "
            f"{r['gpu_count']:6.0f}  {r['topas_count']:7.0f}  "
            f"{r['gpu_mean_product_mevu']:8.2f}  "
            f"{r['topas_mean_product_mevu']:10.2f}  "
            f"{(r['mean_product_mevu_ratio'] or float('nan')):6.3f}"
        )


if __name__ == "__main__":
    main()
