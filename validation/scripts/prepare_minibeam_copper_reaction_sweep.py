#!/usr/bin/env python3
"""Prepare TOPAS Copper C-12 reaction-ntuple cases at multiple energies."""

from __future__ import annotations

import argparse
from pathlib import Path


def case_text(energy_mevu: int, histories: int) -> str:
    total_energy_mev = 12 * energy_mevu
    stem = f"copper_reactions_e{energy_mevu}_{histories}"
    return f"""# Generated Copper C-12 reaction-package energy sweep.
includeFile = run_copper_reactions_2150MeV_20k.txt

i:Ts/Seed = {20260731 + energy_mevu}
i:Ts/NumberOfThreads = 40
i:Ts/ShowHistoryCountAtInterval = {max(1000, histories // 10)}
d:Ph/Default/EMRangeMax = 6 GeV
d:So/CarbonBeam/BeamEnergy = {total_energy_mev} MeV
i:So/CarbonBeam/NumberOfHistoriesInRun = {histories}
s:Sc/CopperReactions/OutputFile = "output/{stem}"
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--energies", type=int, nargs="+", default=[100, 200, 300, 400]
    )
    parser.add_argument("--histories", type=int, default=20_000)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("ct/minibeam")
    )
    args = parser.parse_args()
    if args.histories <= 0 or any(energy <= 0 for energy in args.energies):
        raise ValueError("energies and histories must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for energy in args.energies:
        path = (
            args.output_dir
            / f"run_copper_reactions_e{energy}_{args.histories}.txt"
        )
        path.write_text(
            case_text(energy, args.histories), encoding="utf-8"
        )
        print(path)


if __name__ == "__main__":
    main()
