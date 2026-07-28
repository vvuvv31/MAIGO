#!/usr/bin/env python3
"""Prepare TOPAS minibeam phase-space diagnostics at selected energies."""

from __future__ import annotations

import argparse
from pathlib import Path


def case_text(energy_mevu: int, histories: int) -> str:
    total_energy_mev = 12 * energy_mevu
    stem = f"minibeam_phase_e{energy_mevu}_{histories}"
    return f"""# Generated multi-energy minibeam phase-space diagnostic.
includeFile = run_phase_space_center.txt

i:Ts/Seed = 20260728
i:Ts/NumberOfThreads = 40
i:Ts/ShowHistoryCountAtInterval = {max(1000, histories // 10)}
d:Ph/Default/EMRangeMax = 6 GeV
dv:Tf/Scatterer1/L1/Values = 1 {total_energy_mev} MeV
dv:Tf/Scatterer1/L2/Values = 1 {total_energy_mev} MeV
iv:Tf/Scatterer1/L4/Values = 1 {histories}

s:Sc/AtCollimatorEntrance/OutputFile = "output/{stem}_collimator_entrance"
s:Sc/AtCollimatorExit/OutputFile = "output/{stem}_collimator_exit"
s:Sc/AtWaterEntrance/OutputFile = "output/{stem}_water_entrance"
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--energies", type=int, nargs="+", default=[200, 400]
    )
    parser.add_argument("--histories", type=int, default=10_000)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("ct/minibeam")
    )
    args = parser.parse_args()
    if args.histories <= 0 or any(energy <= 0 for energy in args.energies):
        raise ValueError("energies and histories must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for energy in args.energies:
        output = (
            args.output_dir
            / f"run_phase_space_e{energy}_{args.histories}.txt"
        )
        output.write_text(
            case_text(energy, args.histories), encoding="utf-8"
        )
        print(output)


if __name__ == "__main__":
    main()
