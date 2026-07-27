#!/usr/bin/env python3
"""Histogram TOPAS cascade/primary product CSVs into GPU-compatible birth spectra.

No third-party dependencies (stdlib csv/gzip only).

Writes the same CSV suite as carbon_mc fragment_birth_spectrum_output_file:

  <prefix>_summary.csv
  <prefix>_mevu.csv
  <prefix>_depth.csv
  <prefix>_costheta.csv
  <prefix>_parent_mevu.csv
  <prefix>_parent_z.csv
"""

from __future__ import annotations

import argparse
import csv
import gzip
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Optional, Text, Tuple

SPECIES = ("proton", "deuteron", "triton", "he3", "he4")
SPECIES_ZA = {
    (1, 1): "proton",
    (1, 2): "deuteron",
    (1, 3): "triton",
    (2, 3): "he3",
    (2, 4): "he4",
}
MEVU_BIN_WIDTH = 2.0
MEVU_BINS = 200
COS_BINS = 20
PARENT_MEVU_BIN_WIDTH = 10.0
PARENT_MEVU_BINS = 40
PARENT_Z_BINS = 9
GEN_BINS = 3


def open_rows(path: Path) -> Iterable[Dict[str, str]]:
    if str(path).endswith(".gz"):
        fh = gzip.open(path, "rt", encoding="utf-8", newline="")
    else:
        fh = open(path, "r", encoding="utf-8", newline="")
    with fh:
        reader = csv.DictReader(fh)
        for row in reader:
            yield row


def species_name(z: int, a: int) -> Optional[str]:
    return SPECIES_ZA.get((int(z), int(a)))


def mevu_bin(ke_mev: float, a: int) -> int:
    if a <= 0 or ke_mev <= 0:
        return 0
    mevu = ke_mev / float(a)
    b = int(mevu / MEVU_BIN_WIDTH)
    return min(max(b, 0), MEVU_BINS - 1)


def cos_bin(cz: float) -> int:
    c = max(-1.0, min(1.0, float(cz)))
    b = int(((c + 1.0) * 0.5) * COS_BINS)
    return min(max(b, 0), COS_BINS - 1)


def parent_mevu_bin(ke_mev: float, a: int) -> int:
    if a <= 0 or ke_mev <= 0:
        return 0
    mevu = ke_mev / float(a)
    b = int(mevu / PARENT_MEVU_BIN_WIDTH)
    return min(max(b, 0), PARENT_MEVU_BINS - 1)


def parent_z_bin(z: int) -> int:
    if z <= 0:
        return 0
    return min(int(z), PARENT_Z_BINS - 1)


def write_suite(
    prefix: Path,
    counts_gen: Dict[Tuple[str, int], int],
    ke_sum_gen: Dict[Tuple[str, int], float],
    mevu_hist: Dict[Tuple[str, int], int],
    depth_hist: Dict[Tuple[str, float], int],
    cos_hist: Dict[Tuple[str, int], int],
    parent_mevu_hist: Dict[Tuple[str, int], int],
    parent_z_hist: Dict[Tuple[str, int], int],
    histories: int,
) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{prefix}_summary.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "species",
                "generation",
                "count",
                "mean_kinetic_energy_MeV",
                "yield_per_primary",
            ]
        )
        for sp in SPECIES:
            for gen in range(GEN_BINS):
                c = counts_gen.get((sp, gen), 0)
                ke = ke_sum_gen.get((sp, gen), 0.0)
                mean = ke / c if c else 0.0
                yld = c / float(histories) if histories else 0.0
                w.writerow([sp, gen, c, f"{mean:.12g}", f"{yld:.12g}"])

    with open(f"{prefix}_mevu.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["species", "mevu_bin_low", "mevu_bin_high", "count"])
        for sp in SPECIES:
            for b in range(MEVU_BINS):
                c = mevu_hist.get((sp, b), 0)
                if not c:
                    continue
                low = b * MEVU_BIN_WIDTH
                w.writerow([sp, f"{low:.12g}", f"{low + MEVU_BIN_WIDTH:.12g}", c])

    with open(f"{prefix}_depth.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["species", "depth_mm", "count"])
        for (sp, depth), c in sorted(depth_hist.items()):
            if c:
                w.writerow([sp, f"{depth:.12g}", c])

    with open(f"{prefix}_costheta.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["species", "cos_bin_low", "cos_bin_high", "count"])
        for sp in SPECIES:
            for b in range(COS_BINS):
                c = cos_hist.get((sp, b), 0)
                if not c:
                    continue
                low = -1.0 + 2.0 * b / COS_BINS
                high = -1.0 + 2.0 * (b + 1) / COS_BINS
                w.writerow([sp, f"{low:.12g}", f"{high:.12g}", c])

    with open(f"{prefix}_parent_mevu.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["species", "parent_mevu_bin_low", "parent_mevu_bin_high", "count"]
        )
        for sp in SPECIES:
            for b in range(PARENT_MEVU_BINS):
                c = parent_mevu_hist.get((sp, b), 0)
                if not c:
                    continue
                low = b * PARENT_MEVU_BIN_WIDTH
                w.writerow(
                    [sp, f"{low:.12g}", f"{low + PARENT_MEVU_BIN_WIDTH:.12g}", c]
                )

    with open(f"{prefix}_parent_z.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["species", "parent_Z", "count"])
        for sp in SPECIES:
            for b in range(PARENT_Z_BINS):
                c = parent_z_hist.get((sp, b), 0)
                if c:
                    w.writerow([sp, b, c])


def _int(row: Dict[str, str], *keys: str, default: int = 0) -> int:
    for k in keys:
        if k in row and row[k] != "":
            return int(float(row[k]))
    return default


def _float(row: Dict[str, str], *keys: str, default: float = 0.0) -> float:
    for k in keys:
        if k in row and row[k] != "":
            return float(row[k])
    return default


def process_primary(
    products_path: Path,
    reactions_path: Optional[Path],
    histories: int,
    prefix: Path,
) -> None:
    parent_ke_by_reaction: Dict[int, float] = {}
    if reactions_path is not None:
        for row in open_rows(reactions_path):
            rid = _int(row, "reaction_id", "interaction_id")
            ike = _float(
                row,
                "incident_energy_MeV",
                "incident_kinetic_energy_MeV",
                "projectile_kinetic_energy_MeV",
            )
            parent_ke_by_reaction[rid] = ike

    counts_gen: Dict[Tuple[str, int], int] = defaultdict(int)
    ke_sum_gen: Dict[Tuple[str, int], float] = defaultdict(float)
    mevu_hist: Dict[Tuple[str, int], int] = defaultdict(int)
    depth_hist: Dict[Tuple[str, float], int] = defaultdict(int)
    cos_hist: Dict[Tuple[str, int], int] = defaultdict(int)
    parent_mevu_hist: Dict[Tuple[str, int], int] = defaultdict(int)
    parent_z_hist: Dict[Tuple[str, int], int] = defaultdict(int)

    for row in open_rows(products_path):
        z = _int(row, "atomic_number_Z", "Z")
        a = _int(row, "mass_number_A", "A")
        sp = species_name(z, a)
        if sp is None:
            continue
        ke = _float(row, "kinetic_energy_MeV")
        gen = 0
        counts_gen[(sp, gen)] += 1
        ke_sum_gen[(sp, gen)] += ke
        mevu_hist[(sp, mevu_bin(ke, a))] += 1
        if "direction_z" in row and row["direction_z"] != "":
            cos_hist[(sp, cos_bin(float(row["direction_z"])))] += 1
        rid = _int(row, "reaction_id", "interaction_id", default=-1)
        parent_ke = parent_ke_by_reaction.get(rid, 0.0)
        parent_mevu_hist[(sp, parent_mevu_bin(parent_ke, 12))] += 1
        parent_z_hist[(sp, parent_z_bin(6))] += 1

    write_suite(
        prefix,
        counts_gen,
        ke_sum_gen,
        mevu_hist,
        depth_hist,
        cos_hist,
        parent_mevu_hist,
        parent_z_hist,
        histories,
    )


def process_cascade(
    products_path: Path,
    interactions_path: Optional[Path],
    histories: int,
    prefix: Path,
) -> None:
    parent_by_interaction: Dict[int, Tuple[int, int, float]] = {}
    if interactions_path is not None:
        for row in open_rows(interactions_path):
            iid = _int(row, "interaction_id")
            pz = _int(row, "projectile_z", "projectile_Z")
            pa = _int(row, "projectile_a", "projectile_A")
            ike = _float(row, "incident_energy_mev", "incident_energy_MeV")
            parent_by_interaction[iid] = (pz, pa, ike)

    counts_gen: Dict[Tuple[str, int], int] = defaultdict(int)
    ke_sum_gen: Dict[Tuple[str, int], float] = defaultdict(float)
    mevu_hist: Dict[Tuple[str, int], int] = defaultdict(int)
    depth_hist: Dict[Tuple[str, float], int] = defaultdict(int)
    cos_hist: Dict[Tuple[str, int], int] = defaultdict(int)
    parent_mevu_hist: Dict[Tuple[str, int], int] = defaultdict(int)
    parent_z_hist: Dict[Tuple[str, int], int] = defaultdict(int)

    for row in open_rows(products_path):
        z = _int(row, "Z", "atomic_number_Z")
        a = _int(row, "A", "mass_number_A")
        sp = species_name(z, a)
        if sp is None:
            continue
        ke = _float(row, "kinetic_energy_MeV")
        gen = 1  # package rows treated as first cascade generation
        counts_gen[(sp, gen)] += 1
        ke_sum_gen[(sp, gen)] += ke
        mevu_hist[(sp, mevu_bin(ke, a))] += 1
        if "direction_z" in row and row["direction_z"] != "":
            cos_hist[(sp, cos_bin(float(row["direction_z"])))] += 1
        iid = _int(row, "interaction_id", default=-1)
        if iid in parent_by_interaction:
            pz, pa, pke = parent_by_interaction[iid]
        else:
            pz, pa, pke = 6, 12, 0.0
        parent_mevu_hist[(sp, parent_mevu_bin(pke, pa if pa > 0 else 12))] += 1
        parent_z_hist[(sp, parent_z_bin(pz))] += 1
        if "vertex_z_mm" in row and row["vertex_z_mm"] != "":
            depth_hist[(sp, float(row["vertex_z_mm"]))] += 1

    write_suite(
        prefix,
        counts_gen,
        ke_sum_gen,
        mevu_hist,
        depth_hist,
        cos_hist,
        parent_mevu_hist,
        parent_z_hist,
        histories,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--products", type=Path, required=True)
    parser.add_argument("--interactions", type=Path, default=None)
    parser.add_argument("--reactions", type=Path, default=None)
    parser.add_argument(
        "--source",
        choices=("primary", "cascade"),
        required=True,
    )
    parser.add_argument("--histories", type=int, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    if args.source == "primary":
        process_primary(
            args.products, args.reactions, args.histories, args.output_prefix
        )
    else:
        process_cascade(
            args.products, args.interactions, args.histories, args.output_prefix
        )
    print(f"Wrote birth spectrum suite under {args.output_prefix}_*.csv")


if __name__ == "__main__":
    main()
