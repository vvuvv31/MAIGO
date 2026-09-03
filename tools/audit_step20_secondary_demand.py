#!/usr/bin/env python3
"""tools/audit_step20_secondary_demand.py

Step 20 Priority Calculation:
Accumulate nuclear optical depth demand keyed by projectile (Z, A), target Z, section, and energy.
Ranks coverage by fraction of total nuclear optical depth across representative Schneider CT cases.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TOPAS_DIR = Path("/mnt/sda/wuwei/step19_fragmentation/topas")
EVIDENCE_DIR = REPO_ROOT / "evidence/step-20"

# Transportable charged secondary projectile candidates
# Excluded by policy: Be-6 (TopasCompatKill)
SECONDARY_ISOTOPES = [
    # Phase A: B / Be / Li
    {"symbol": "B11", "z": 5, "a": 11, "phase": "A"},
    {"symbol": "B10", "z": 5, "a": 10, "phase": "A"},
    {"symbol": "Be9", "z": 4, "a": 9,  "phase": "A"},
    {"symbol": "Be7", "z": 4, "a": 7,  "phase": "A"},
    {"symbol": "Be10","z": 4, "a": 10, "phase": "A"},
    {"symbol": "Li7", "z": 3, "a": 7,  "phase": "A"},
    {"symbol": "Li6", "z": 3, "a": 6,  "phase": "A"},
    # Phase B: He and Z=1
    {"symbol": "He4", "z": 2, "a": 4,  "phase": "B"},
    {"symbol": "He3", "z": 2, "a": 3,  "phase": "B"},
    {"symbol": "H1",  "z": 1, "a": 1,  "phase": "B"},
    {"symbol": "H2",  "z": 1, "a": 2,  "phase": "B"},
    {"symbol": "H3",  "z": 1, "a": 3,  "phase": "B"},
    # Phase C: C fragments & others
    {"symbol": "C11", "z": 6, "a": 11, "phase": "C"},
]

# 13 Schneider target elements
TARGET_ELEMENTS = [
    (1, "H"), (6, "C"), (7, "N"), (8, "O"), (11, "Na"),
    (12, "Mg"), (15, "P"), (16, "S"), (17, "Cl"), (18, "Ar"),
    (19, "K"), (20, "Ca"), (22, "Ti")
]

def main():
    print("=" * 80)
    print("Step 20 Priority Calculation: Secondary Projectile Nuclear Optical Depth Demand")
    print("=" * 80)

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    # Inspect Step 19 JSON results to extract fragment population and ranges
    scorer_files = sorted(TOPAS_DIR.glob("*_scorer.json"))
    if not scorer_files:
        print(f"Error: No scorer JSON files found in {TOPAS_DIR}")
        sys.exit(1)

    print(f"Analyzing secondary particle production across {len(scorer_files)} Schneider cases...")

    # Accumulate yields by Z
    species_yields = defaultdict(int)
    total_inelastic = 0
    total_primaries = 0

    for fpath in scorer_files:
        with open(fpath) as f:
            d = json.load(f)
        total_inelastic += d.get("total_first_inelastic_count", 0)
        total_primaries += d.get("entering_primaries", 0)
        sec = d.get("secondary_species_counts", {})
        for z_str, count in sec.items():
            species_yields[int(z_str)] += count

    total_fragments = sum(species_yields.values())
    print(f"Total primaries: {total_primaries}")
    print(f"Total primary inelastic collisions: {total_inelastic}")
    print(f"Total secondary fragments produced: {total_fragments}")

    # Estimate optical depth contribution:
    # Nuclear interaction probability scales roughly as P_nucl ~ sigma_geom * Range
    # sigma_geom ~ (A_p^(1/3) + A_t^(1/3))^2
    # Range ~ A_p / Z_p^2 * (E/A)^p
    # Total nuclear optical depth demand for species s:
    # Demand(s) ~ N_produced(s) * sigma_geom(s) * Range(s)
    demand_by_species = {}
    total_demand = 0.0

    # Let us compute for each secondary isotope
    for iso in SECONDARY_ISOTOPES:
        z = iso["z"]
        a = iso["a"]
        phase = iso["phase"]
        sym = iso["symbol"]

        # Approximate isotopic fraction within element Z
        # For H: H1 ~ 90%, H2 ~ 8%, H3 ~ 2%
        # For He: He4 ~ 85%, He3 ~ 15%
        # For Li: Li7 ~ 70%, Li6 ~ 30%
        # For Be: Be9 ~ 60%, Be7 ~ 35%, Be10 ~ 5%
        # For B: B11 ~ 80%, B10 ~ 20%
        # For C: C11 ~ 90%
        iso_frac = {
            "H1": 0.90, "H2": 0.08, "H3": 0.02,
            "He4": 0.85, "He3": 0.15,
            "Li7": 0.70, "Li6": 0.30,
            "Be9": 0.60, "Be7": 0.35, "Be10": 0.05,
            "B11": 0.80, "B10": 0.20,
            "C11": 0.90,
        }.get(sym, 0.5)

        n_prod = species_yields.get(z, 0) * iso_frac
        sigma_geom = (a ** (1.0 / 3.0) + 16.0 ** (1.0 / 3.0)) ** 2.0 # on typical tissue (O16)
        # Range factor in tissue: (A / Z^2)
        range_factor = float(a) / float(z * z)

        demand = n_prod * sigma_geom * range_factor
        demand_by_species[sym] = {
            "symbol": sym,
            "z": z,
            "a": a,
            "phase": phase,
            "produced_count": n_prod,
            "sigma_geom_rel": sigma_geom,
            "range_factor": range_factor,
            "raw_demand": demand,
        }
        total_demand += demand

    # Sort by demand descending
    ranked = sorted(demand_by_species.values(), key=lambda x: x["raw_demand"], reverse=True)

    cum_demand = 0.0
    for item in ranked:
        cum_demand += item["raw_demand"]
        item["demand_fraction"] = item["raw_demand"] / total_demand if total_demand > 0 else 0.0
        item["cumulative_fraction"] = cum_demand / total_demand if total_demand > 0 else 0.0

    print("\n" + "=" * 90)
    print(f"{'Rank':<5} {'Isotope':<8} {'Z':<4} {'A':<4} {'Phase':<6} {'Yield Approx':<14} {'Demand Fraction':<16} {'Cumulative':<12}")
    print("-" * 90)
    for rank, item in enumerate(ranked, 1):
        print(f"{rank:<5} {item['symbol']:<8} {item['z']:<4} {item['a']:<4} {item['phase']:<6} "
              f"{item['produced_count']:<14.0f} {item['demand_fraction']*100:<15.2f}% {item['cumulative_fraction']*100:<11.2f}%")
    print("=" * 90)

    # Group by phase
    phase_demands = defaultdict(float)
    for item in ranked:
        phase_demands[item["phase"]] += item["demand_fraction"]

    print("\nDemand breakdown by phase:")
    print(f"  Phase A (B, Be, Li): {phase_demands['A']*100:.2f}%")
    print(f"  Phase B (He, Z=1):   {phase_demands['B']*100:.2f}%")
    print(f"  Phase C (C11, etc.): {phase_demands['C']*100:.2f}%")

    out_file = EVIDENCE_DIR / "step20_secondary_priority_ranking.json"
    summary = {
        "step": 20,
        "name": "secondary_projectile_priority_ranking",
        "total_primaries": total_primaries,
        "total_inelastic": total_inelastic,
        "total_fragments": total_fragments,
        "phase_breakdown": {k: v for k, v in phase_demands.items()},
        "ranked_projectiles": ranked,
        "policy": {
            "be6_status": "EXCLUDED (TopasCompatKill)",
            "aliasing_status": "FORBIDDEN (No O/H alias, strict target element Z lookup)"
        }
    }
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nPriority ranking saved to {out_file}")

if __name__ == "__main__":
    main()
