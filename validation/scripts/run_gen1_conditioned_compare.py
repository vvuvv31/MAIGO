#!/usr/bin/env python3
"""Build projectile/energy-filtered TOPAS cascade refs and compare GPU gen1.

Produces several conditioned TOPAS prefixes and JSON comparisons against a GPU
birth-spectrum prefix (must include generation-split CSVs).

Example:
  python3 validation/scripts/run_gen1_conditioned_compare.py \\
    --gpu-prefix validation/results/birth_spectrum_energy_suite/gpu_e400 \\
    --output-dir validation/results/birth_spectrum_energy_suite/gen1_conditioned
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PRODUCTS = (
    ROOT
    / "validation/results/topas_400MeVu_cascade_g4_11_3_2_100k_products.csv.gz"
)
DEFAULT_INTERACTIONS = (
    ROOT
    / "validation/results/topas_400MeVu_cascade_g4_11_3_2_100k_interactions.csv.gz"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-prefix", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--products", type=Path, default=DEFAULT_PRODUCTS)
    parser.add_argument("--interactions", type=Path, default=DEFAULT_INTERACTIONS)
    parser.add_argument("--histories", type=int, default=100000)
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    prepare = ROOT / "validation/scripts/prepare_topas_birth_spectrum.py"
    compare = ROOT / "validation/scripts/compare_fragment_birth_spectra.py"

    cases = [
        {
            "name": "all_cascade",
            "args": [],
            "note": "All cascade package light products labeled gen=1",
        },
        {
            "name": "parent_C12",
            "args": ["--projectile-z", "6", "--projectile-a", "12"],
            "note": "Only products of C-12 projectiles (secondary C12 re-reactions)",
        },
        {
            "name": "parent_fragment_Zle5",
            "args": ["--parent-z-max", "5"],
            "note": "Parents with Z<=5 (typical charged fragments, not C)",
        },
        {
            "name": "parent_he4",
            "args": ["--projectile-z", "2", "--projectile-a", "4"],
            "note": "He-4 projectile cascade products only",
        },
        {
            "name": "parent_mevu_0_100",
            "args": ["--parent-mevu-min", "0", "--parent-mevu-max", "100"],
            "note": "Parent incident energy 0-100 MeV/u",
        },
        {
            "name": "parent_mevu_100_250",
            "args": ["--parent-mevu-min", "100", "--parent-mevu-max", "250"],
            "note": "Parent incident energy 100-250 MeV/u",
        },
        {
            "name": "parent_mevu_250_400",
            "args": ["--parent-mevu-min", "250", "--parent-mevu-max", "400"],
            "note": "Parent incident energy 250-400 MeV/u",
        },
    ]

    report = {"gpu_prefix": str(args.gpu_prefix), "cases": {}}
    for case in cases:
        prefix = out / f"topas_{case['name']}"
        cmd = [
            sys.executable,
            str(prepare),
            "--products",
            str(args.products),
            "--interactions",
            str(args.interactions),
            "--source",
            "cascade",
            "--histories",
            str(args.histories),
            "--generation",
            "1",
            "--output-prefix",
            str(prefix),
            *case["args"],
        ]
        print("Running:", " ".join(cmd), flush=True)
        subprocess.check_call(cmd)
        cmp_json = out / f"compare_gen1_vs_{case['name']}.json"
        subprocess.check_call(
            [
                sys.executable,
                str(compare),
                "--gpu-prefix",
                str(args.gpu_prefix),
                "--topas-prefix",
                str(prefix),
                "--generation",
                "1",
                "--output-json",
                str(cmp_json),
            ]
        )
        with open(cmp_json, encoding="utf-8") as f:
            cmp = json.load(f)
        report["cases"][case["name"]] = {
            "note": case["note"],
            "topas_prefix": str(prefix),
            "compare_json": str(cmp_json),
            "by_species": cmp["by_species"],
        }

    summary = out / "gen1_conditioned_summary.json"
    with open(summary, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {summary}")

    print(
        "\ncase                    he4_yield_ratio  he4_meanKE_ratio  "
        "he4_mean_mevu_ratio  he4_parent_mevu_ratio"
    )
    for name, rec in report["cases"].items():
        h = rec["by_species"]["he4"]
        print(
            f"{name:24s}  {h.get('yield_ratio_gpu_over_topas'):8.3f}  "
            f"{h.get('mean_ke_ratio_gpu_over_topas'):8.3f}  "
            f"{h.get('mean_mevu_ratio'):8.3f}  "
            f"{h.get('mean_parent_mevu_ratio'):8.3f}"
        )


if __name__ == "__main__":
    main()
