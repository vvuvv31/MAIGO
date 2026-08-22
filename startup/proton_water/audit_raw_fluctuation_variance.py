#!/usr/bin/env python3
"""Report raw TOPAS thin-slab loss moments for selected campaign points."""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from topas_ascii_ntuple import read_energy_loss_fluctuation_ntuple


ENERGIES = (70, 100, 150, 200, 250)
DENSITY_LABELS = ("0p0003125", "0p000625", "0p00125", "0p0025")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_root", type=Path)
    args = parser.parse_args()
    print("energy_MeV,density_label,samples,mean_loss_MeV,normalized_variance")
    for energy in ENERGIES:
        for density_label in DENSITY_LABELS:
            path = (
                args.campaign_root
                / "runs"
                / f"e{energy}mevu_ad{density_label}gcm2"
                / "energy_loss_fluctuation.phsp"
            )
            _, rows = read_energy_loss_fluctuation_ntuple(path)
            losses = [row.kinetic_energy_loss_mev for row in rows]
            mean = statistics.fmean(losses)
            variance = statistics.fmean((loss / mean - 1.0) ** 2 for loss in losses)
            print(
                f"{energy},{density_label},{len(losses)},"
                f"{mean:.12g},{variance:.12g}"
            )


if __name__ == "__main__":
    main()
