#!/usr/bin/env python3
"""Histogram TOPAS cascade/primary product CSVs into GPU-compatible birth spectra.

Stdlib only. Histogram CSVs include a generation column matching GPU output:

  species,generation,mevu_bin_low,mevu_bin_high,count
  species,generation,parent_mevu_bin_low,parent_mevu_bin_high,count

Cascade products default to generation=1; primary secondaries to generation=0.

Optional filters for conditioned gen1 references:
  --parent-z-min/max, --parent-mevu-min/max, --projectile-z/a
"""

from __future__ import annotations

import argparse
import csv
import gzip
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

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
DEPTH_BIN_WIDTH_MM = 1.0


def open_rows(path: Path) -> Iterable[Dict[str, str]]:
    if str(path).endswith(".gz"):
        fh = gzip.open(path, "rt", encoding="utf-8", newline="")
    else:
        fh = open(path, "r", encoding="utf-8", newline="")
    with fh:
        yield from csv.DictReader(fh)


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


def depth_bin_center(depth_mm: float) -> float:
    """Match the 1 mm birth-depth histogram used by the GPU diagnostics."""
    index = int(max(float(depth_mm), 0.0) / DEPTH_BIN_WIDTH_MM)
    return (index + 0.5) * DEPTH_BIN_WIDTH_MM


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


def write_suite(
    prefix: Path,
    counts_gen: Dict[Tuple[str, int], int],
    ke_sum_gen: Dict[Tuple[str, int], float],
    mevu_hist: Dict[Tuple[str, int, int], int],
    depth_hist: Dict[Tuple[str, int, float], int],
    cos_hist: Dict[Tuple[str, int, int], int],
    parent_mevu_hist: Dict[Tuple[str, int, int], int],
    parent_z_hist: Dict[Tuple[str, int, int], int],
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
        w.writerow(
            ["species", "generation", "mevu_bin_low", "mevu_bin_high", "count"]
        )
        for sp in SPECIES:
            for gen in range(GEN_BINS):
                for b in range(MEVU_BINS):
                    c = mevu_hist.get((sp, gen, b), 0)
                    if not c:
                        continue
                    low = b * MEVU_BIN_WIDTH
                    w.writerow(
                        [sp, gen, f"{low:.12g}", f"{low + MEVU_BIN_WIDTH:.12g}", c]
                    )

    with open(f"{prefix}_depth.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["species", "generation", "depth_mm", "count"])
        for (sp, gen, depth), c in sorted(depth_hist.items()):
            if c:
                w.writerow([sp, gen, f"{depth:.12g}", c])

    with open(f"{prefix}_costheta.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["species", "generation", "cos_bin_low", "cos_bin_high", "count"]
        )
        for sp in SPECIES:
            for gen in range(GEN_BINS):
                for b in range(COS_BINS):
                    c = cos_hist.get((sp, gen, b), 0)
                    if not c:
                        continue
                    low = -1.0 + 2.0 * b / COS_BINS
                    high = -1.0 + 2.0 * (b + 1) / COS_BINS
                    w.writerow([sp, gen, f"{low:.12g}", f"{high:.12g}", c])

    with open(f"{prefix}_parent_mevu.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "species",
                "generation",
                "parent_mevu_bin_low",
                "parent_mevu_bin_high",
                "count",
            ]
        )
        for sp in SPECIES:
            for gen in range(GEN_BINS):
                for b in range(PARENT_MEVU_BINS):
                    c = parent_mevu_hist.get((sp, gen, b), 0)
                    if not c:
                        continue
                    low = b * PARENT_MEVU_BIN_WIDTH
                    w.writerow(
                        [
                            sp,
                            gen,
                            f"{low:.12g}",
                            f"{low + PARENT_MEVU_BIN_WIDTH:.12g}",
                            c,
                        ]
                    )

    with open(f"{prefix}_parent_z.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["species", "generation", "parent_Z", "count"])
        for sp in SPECIES:
            for gen in range(GEN_BINS):
                for b in range(PARENT_Z_BINS):
                    c = parent_z_hist.get((sp, gen, b), 0)
                    if c:
                        w.writerow([sp, gen, b, c])


def process_primary(
    products_path: Path,
    reactions_path: Optional[Path],
    histories: int,
    prefix: Path,
    generation: int,
) -> None:
    parent_by_reaction: Dict[int, Tuple[float, float]] = {}
    if reactions_path is not None:
        for row in open_rows(reactions_path):
            rid = _int(row, "reaction_id", "interaction_id")
            ike = _float(
                row,
                "incident_energy_MeV",
                "incident_kinetic_energy_MeV",
                "projectile_kinetic_energy_MeV",
            )
            depth = _float(row, "depth_mm", "reaction_depth_mm", "vertex_z_mm")
            parent_by_reaction[rid] = (ike, depth)

    counts_gen: Dict[Tuple[str, int], int] = defaultdict(int)
    ke_sum_gen: Dict[Tuple[str, int], float] = defaultdict(float)
    mevu_hist: Dict[Tuple[str, int, int], int] = defaultdict(int)
    depth_hist: Dict[Tuple[str, int, float], int] = defaultdict(int)
    cos_hist: Dict[Tuple[str, int, int], int] = defaultdict(int)
    parent_mevu_hist: Dict[Tuple[str, int, int], int] = defaultdict(int)
    parent_z_hist: Dict[Tuple[str, int, int], int] = defaultdict(int)

    for row in open_rows(products_path):
        z = _int(row, "atomic_number_Z", "Z")
        a = _int(row, "mass_number_A", "A")
        sp = species_name(z, a)
        if sp is None:
            continue
        ke = _float(row, "kinetic_energy_MeV")
        gen = generation
        counts_gen[(sp, gen)] += 1
        ke_sum_gen[(sp, gen)] += ke
        mevu_hist[(sp, gen, mevu_bin(ke, a))] += 1
        if "direction_z" in row and row["direction_z"] != "":
            cos_hist[(sp, gen, cos_bin(float(row["direction_z"])))] += 1
        rid = _int(row, "reaction_id", "interaction_id", default=-1)
        parent_ke, depth = parent_by_reaction.get(rid, (0.0, 0.0))
        parent_mevu_hist[(sp, gen, parent_mevu_bin(parent_ke, 12))] += 1
        parent_z_hist[(sp, gen, parent_z_bin(6))] += 1
        if reactions_path is not None and rid in parent_by_reaction:
            depth_hist[(sp, gen, depth_bin_center(depth))] += 1

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
    generation: int,
    parent_z_min: Optional[int],
    parent_z_max: Optional[int],
    parent_mevu_min: Optional[float],
    parent_mevu_max: Optional[float],
    projectile_z: Optional[int],
    projectile_a: Optional[int],
) -> Dict[str, int]:
    parent_by_interaction: Dict[int, Tuple[int, int, float, float]] = {}
    if interactions_path is not None:
        for row in open_rows(interactions_path):
            iid = _int(row, "interaction_id")
            pz = _int(row, "projectile_z", "projectile_Z")
            pa = _int(row, "projectile_a", "projectile_A")
            ike = _float(
                row,
                "incident_energy_MeV_per_u",
                "incident_energy_mev_per_u",
                "incident_energy_MeV",
            )
            # Prefer MeV/u when present; if only total MeV given, convert later.
            if "incident_energy_MeV_per_u" not in row and "incident_energy_mev_per_u" not in row:
                if pa > 0 and ike > 0:
                    # column was total MeV
                    ike = ike / float(pa)
            depth = _float(row, "depth_mm", "vertex_z_mm")
            parent_by_interaction[iid] = (pz, pa, ike, depth)

    counts_gen: Dict[Tuple[str, int], int] = defaultdict(int)
    ke_sum_gen: Dict[Tuple[str, int], float] = defaultdict(float)
    mevu_hist: Dict[Tuple[str, int, int], int] = defaultdict(int)
    depth_hist: Dict[Tuple[str, int, float], int] = defaultdict(int)
    cos_hist: Dict[Tuple[str, int, int], int] = defaultdict(int)
    parent_mevu_hist: Dict[Tuple[str, int, int], int] = defaultdict(int)
    parent_z_hist: Dict[Tuple[str, int, int], int] = defaultdict(int)

    kept = 0
    skipped = 0
    for row in open_rows(products_path):
        z = _int(row, "Z", "atomic_number_Z")
        a = _int(row, "A", "mass_number_A")
        sp = species_name(z, a)
        if sp is None:
            continue
        iid = _int(row, "interaction_id", default=-1)
        if iid in parent_by_interaction:
            pz, pa, pmevu, parent_depth = parent_by_interaction[iid]
            # If stored as total MeV under wrong key, repair when pe is large and A known
            if pmevu > 1000 and pa > 0:
                pmevu = pmevu / float(pa)
        else:
            pz, pa, pmevu, parent_depth = 0, 0, 0.0, 0.0

        if projectile_z is not None and pz != projectile_z:
            skipped += 1
            continue
        if projectile_a is not None and pa != projectile_a:
            skipped += 1
            continue
        if parent_z_min is not None and pz < parent_z_min:
            skipped += 1
            continue
        if parent_z_max is not None and pz > parent_z_max:
            skipped += 1
            continue
        if parent_mevu_min is not None and pmevu < parent_mevu_min:
            skipped += 1
            continue
        if parent_mevu_max is not None and pmevu > parent_mevu_max:
            skipped += 1
            continue

        ke = _float(row, "kinetic_energy_MeV")
        gen = generation
        counts_gen[(sp, gen)] += 1
        ke_sum_gen[(sp, gen)] += ke
        mevu_hist[(sp, gen, mevu_bin(ke, a))] += 1
        if "direction_z" in row and row["direction_z"] != "":
            cos_hist[(sp, gen, cos_bin(float(row["direction_z"])))] += 1
        # parent_mevu_bin expects total KE and A; convert MeV/u back to total.
        parent_total_ke = pmevu * float(pa) if pa > 0 else 0.0
        parent_mevu_hist[(sp, gen, parent_mevu_bin(parent_total_ke, pa if pa > 0 else 12))] += 1
        parent_z_hist[(sp, gen, parent_z_bin(pz))] += 1
        if iid in parent_by_interaction:
            depth_hist[(sp, gen, depth_bin_center(parent_depth))] += 1
        elif "vertex_z_mm" in row and row["vertex_z_mm"] != "":
            depth_hist[
                (sp, gen, depth_bin_center(float(row["vertex_z_mm"])))
            ] += 1
        kept += 1

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
    return {"kept": kept, "skipped": skipped}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--products", type=Path, required=True)
    parser.add_argument("--interactions", type=Path, default=None)
    parser.add_argument("--reactions", type=Path, default=None)
    parser.add_argument(
        "--source", choices=("primary", "cascade"), required=True
    )
    parser.add_argument("--histories", type=int, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument(
        "--generation",
        type=int,
        default=None,
        help="Generation label written to CSVs (default: 0 primary, 1 cascade)",
    )
    parser.add_argument("--parent-z-min", type=int, default=None)
    parser.add_argument("--parent-z-max", type=int, default=None)
    parser.add_argument("--parent-mevu-min", type=float, default=None)
    parser.add_argument("--parent-mevu-max", type=float, default=None)
    parser.add_argument("--projectile-z", type=int, default=None)
    parser.add_argument("--projectile-a", type=int, default=None)
    args = parser.parse_args()

    if args.source == "primary":
        gen = 0 if args.generation is None else args.generation
        process_primary(
            args.products, args.reactions, args.histories, args.output_prefix, gen
        )
        print(f"Wrote birth spectrum suite under {args.output_prefix}_*.csv (gen={gen})")
    else:
        gen = 1 if args.generation is None else args.generation
        stats = process_cascade(
            args.products,
            args.interactions,
            args.histories,
            args.output_prefix,
            gen,
            args.parent_z_min,
            args.parent_z_max,
            args.parent_mevu_min,
            args.parent_mevu_max,
            args.projectile_z,
            args.projectile_a,
        )
        print(
            f"Wrote birth spectrum suite under {args.output_prefix}_*.csv "
            f"(gen={gen}, kept={stats['kept']}, skipped={stats['skipped']})"
        )


if __name__ == "__main__":
    main()
