#!/usr/bin/env python3
"""Compare GPU vs TOPAS light-isotope birth spectra (futureStep A.4).

Supports generation-split mevu/parent_mevu CSVs. Use --generation to restrict
yield, mean KE, and MeV/u histogram comparisons.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

SPECIES = ("proton", "deuteron", "triton", "he3", "he4")
MEVU_BINS = 200
MEVU_WIDTH = 2.0


def load_summary(prefix: Path) -> List[Dict[str, str]]:
    with open(f"{prefix}_summary.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_mevu(prefix: Path) -> List[Dict[str, str]]:
    path = Path(f"{prefix}_mevu.csv")
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_parent_mevu(prefix: Path) -> List[Dict[str, str]]:
    path = Path(f"{prefix}_parent_mevu.csv")
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def total_yield(
    rows: List[Dict[str, str]],
    species: str,
    generation: Optional[int] = None,
) -> float:
    total = 0.0
    for r in rows:
        if r["species"] != species:
            continue
        if generation is not None and int(r["generation"]) != generation:
            continue
        total += float(r["yield_per_primary"])
    return total


def mean_ke(
    rows: List[Dict[str, str]],
    species: str,
    generation: Optional[int] = None,
) -> float:
    num = 0.0
    den = 0.0
    for r in rows:
        if r["species"] != species:
            continue
        if generation is not None and int(r["generation"]) != generation:
            continue
        c = float(r["count"])
        num += c * float(r["mean_kinetic_energy_MeV"])
        den += c
    return num / den if den > 0 else 0.0


def mevu_hist(
    rows: List[Dict[str, str]],
    species: str,
    generation: Optional[int] = None,
) -> List[float]:
    hist = [0.0] * MEVU_BINS
    for r in rows:
        if r["species"] != species:
            continue
        if generation is not None and "generation" in r and r["generation"] != "":
            if int(r["generation"]) != generation:
                continue
        b = int(round(float(r["mevu_bin_low"]) / MEVU_WIDTH))
        if 0 <= b < MEVU_BINS:
            hist[b] += float(r["count"])
    return hist


def parent_mevu_profile(
    rows: List[Dict[str, str]],
    species: str,
    generation: Optional[int] = None,
) -> Dict[str, float]:
    """Return mean parent MeV/u and total counts for species/generation."""
    num = 0.0
    den = 0.0
    for r in rows:
        if r["species"] != species:
            continue
        if generation is not None and "generation" in r and r["generation"] != "":
            if int(r["generation"]) != generation:
                continue
        c = float(r["count"])
        low = float(r["parent_mevu_bin_low"])
        high = float(r["parent_mevu_bin_high"])
        center = 0.5 * (low + high)
        num += c * center
        den += c
    return {
        "count": den,
        "mean_parent_mevu": num / den if den > 0 else 0.0,
    }


def hist_metrics(gpu: List[float], ref: List[float]) -> Dict[str, float]:
    gsum = sum(gpu)
    rsum = sum(ref)
    g = [x / gsum for x in gpu] if gsum > 0 else [0.0] * len(gpu)
    r = [x / rsum for x in ref] if rsum > 0 else [0.0] * len(ref)
    l1 = sum(abs(a - b) for a, b in zip(g, r))
    centers = [(i + 0.5) * MEVU_WIDTH for i in range(len(g))]
    mean_g = sum(a * c for a, c in zip(g, centers)) if gsum > 0 else 0.0
    mean_r = sum(a * c for a, c in zip(r, centers)) if rsum > 0 else 0.0
    ratio = mean_g / mean_r if mean_r > 0 else float("nan")
    return {
        "hist_l1": l1,
        "mean_mevu_gpu": mean_g,
        "mean_mevu_topas": mean_r,
        "mean_mevu_ratio": ratio,
        "count_gpu": gsum,
        "count_topas": rsum,
    }


def safe_ratio(a: float, b: float) -> float:
    return a / b if b > 0 else float("nan")


def compare(
    gpu_prefix: Path,
    topas_prefix: Path,
    generation: Optional[int] = None,
) -> Dict:
    gpu_sum = load_summary(gpu_prefix)
    top_sum = load_summary(topas_prefix)
    gpu_mevu = load_mevu(gpu_prefix)
    top_mevu = load_mevu(topas_prefix)
    gpu_parent = load_parent_mevu(gpu_prefix)
    top_parent = load_parent_mevu(topas_prefix)

    by_species = {}
    for sp in SPECIES:
        gy = total_yield(gpu_sum, sp, generation)
        ty = total_yield(top_sum, sp, generation)
        gk = mean_ke(gpu_sum, sp, generation)
        tk = mean_ke(top_sum, sp, generation)
        hm = hist_metrics(
            mevu_hist(gpu_mevu, sp, generation),
            mevu_hist(top_mevu, sp, generation),
        )
        gp = parent_mevu_profile(gpu_parent, sp, generation)
        tp = parent_mevu_profile(top_parent, sp, generation)
        by_species[sp] = {
            "yield_per_primary_gpu": gy,
            "yield_per_primary_topas": ty,
            "yield_ratio_gpu_over_topas": safe_ratio(gy, ty),
            "mean_ke_MeV_gpu": gk,
            "mean_ke_MeV_topas": tk,
            "mean_ke_ratio_gpu_over_topas": safe_ratio(gk, tk),
            "mean_parent_mevu_gpu": gp["mean_parent_mevu"],
            "mean_parent_mevu_topas": tp["mean_parent_mevu"],
            "mean_parent_mevu_ratio": safe_ratio(
                gp["mean_parent_mevu"], tp["mean_parent_mevu"]
            ),
            **hm,
        }

    gen_rows = []
    for sp in SPECIES:
        for gen in (0, 1, 2):
            gy = sum(
                float(r["yield_per_primary"])
                for r in gpu_sum
                if r["species"] == sp and int(r["generation"]) == gen
            )
            ty = sum(
                float(r["yield_per_primary"])
                for r in top_sum
                if r["species"] == sp and int(r["generation"]) == gen
            )
            gen_rows.append(
                {
                    "species": sp,
                    "generation": gen,
                    "gpu_yield": gy,
                    "topas_yield": ty,
                }
            )

    return {
        "gpu_prefix": str(gpu_prefix),
        "topas_prefix": str(topas_prefix),
        "generation_filter": generation,
        "by_species": by_species,
        "by_generation": gen_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-prefix", type=Path, required=True)
    parser.add_argument("--topas-prefix", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--generation", type=int, default=None)
    args = parser.parse_args()

    report = compare(args.gpu_prefix, args.topas_prefix, args.generation)

    def sanitize(obj):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        return obj

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(sanitize(report), f, indent=2)

    print(f"Wrote {args.output_json}")
    print(
        "species  yield_ratio  mean_ke_ratio  hist_L1  mean_mevu_ratio  "
        "parent_mevu_ratio"
    )
    for sp, m in report["by_species"].items():
        print(
            f"{sp:8s}  {m['yield_ratio_gpu_over_topas']:8.3f}  "
            f"{m['mean_ke_ratio_gpu_over_topas']:8.3f}  "
            f"{m['hist_l1']:7.3f}  {m['mean_mevu_ratio']:8.3f}  "
            f"{m['mean_parent_mevu_ratio']:8.3f}"
        )


if __name__ == "__main__":
    main()
