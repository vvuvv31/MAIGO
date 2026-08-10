#!/usr/bin/env python3
"""Generate one TOPAS job that queries all Schneider material XS tables.

Every material is defined at 1 g/cm3, so the reported macroscopic cross
section is also its mass cross section in cm3/g up to the common unit factor.
The companion collector can therefore form version-robust ratios to water.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schneider_hu import load_schneider_table


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("schneider")
    parser.add_argument("output")
    parser.add_argument("--result-prefix", default="/tmp/maigo_schneider_xs")
    parser.add_argument("--threads", type=int, default=25)
    args = parser.parse_args()

    table = load_schneider_table(Path(args.schneider))
    if table.material_weights is None or not table.elements:
        raise SystemExit("Schneider table has no material compositions")

    lines = [
        "i:Ts/Seed = 20260723",
        f"i:Ts/NumberOfThreads = {max(1, args.threads)}",
        'b:Ts/ShowHistoryCountLessFrequentlyAsSimulationProgresses = "True"',
        'b:Gr/Enable = "False"',
        'b:Ge/CheckForUnusedComponents = "False"',
        "",
        's:Ge/World/Type = "TsBox"',
        's:Ge/World/Material = "Air"',
        "d:Ge/World/HLX = 10. cm",
        "d:Ge/World/HLY = 2. cm",
        "d:Ge/World/HLZ = 2. cm",
        "",
    ]
    for section in range(table.n_material_sections):
        weights = table.material_weights[section]
        members = [(name, float(weight)) for name, weight in zip(table.elements, weights)
                   if weight > 0.0]
        names = " ".join(f'"{name}"' for name, _ in members)
        fractions = " ".join(f"{weight:.9g}" for _, weight in members)
        x_mm = (section - (table.n_material_sections - 1) / 2.0) * 6.0
        mat = f"SchneiderXS{section:02d}"
        sample = f"Sample{section:02d}"
        beam_position = f"BeamPosition{section:02d}"
        source = f"CarbonBeam{section:02d}"
        scorer = f"CarbonCrossSections{section:02d}"
        lines.extend([
            f"sv:Ma/{mat}/Components = {len(members)} {names}",
            f"uv:Ma/{mat}/Fractions = {len(members)} {fractions}",
            f"d:Ma/{mat}/Density = 1.0 g/cm3",
            f's:Ge/{sample}/Parent = "World"',
            f's:Ge/{sample}/Type = "TsBox"',
            f's:Ge/{sample}/Material = "{mat}"',
            f"d:Ge/{sample}/HLX = 2.0 mm",
            f"d:Ge/{sample}/HLY = 2.0 mm",
            f"d:Ge/{sample}/HLZ = 2.0 mm",
            f"d:Ge/{sample}/TransX = {x_mm:.6g} mm",
            f's:Ge/{beam_position}/Parent = "World"',
            f's:Ge/{beam_position}/Type = "Group"',
            f"d:Ge/{beam_position}/TransX = {x_mm:.6g} mm",
            f"d:Ge/{beam_position}/TransZ = -5.0 mm",
            f's:So/{source}/Type = "Beam"',
            f's:So/{source}/Component = "{beam_position}"',
            f's:So/{source}/BeamParticle = "GenericIon(6,12)"',
            f"d:So/{source}/BeamEnergy = 2400. MeV",
            f"u:So/{source}/BeamEnergySpread = 0.0",
            f's:So/{source}/BeamPositionDistribution = "None"',
            f's:So/{source}/BeamAngularDistribution = "None"',
            f"i:So/{source}/NumberOfHistoriesInRun = 1",
            f's:Sc/{scorer}/Quantity = "CarbonCrossSectionNtuple"',
            f's:Sc/{scorer}/Component = "{sample}"',
            f's:Sc/{scorer}/OutputFile = "{args.result_prefix}_{section:02d}"',
            f's:Sc/{scorer}/OutputType = "ASCII"',
            f's:Sc/{scorer}/IfOutputFileAlreadyExists = "Overwrite"',
            "",
        ])
    lines.extend([
        's:Ph/ListName = "Default"',
        's:Ph/Default/Type = "Geant4_Modular"',
        'sv:Ph/Default/Modules = 6 "g4em-standard_opt4" "g4h-phy_QGSP_BIC_HP" '
        '"g4decay" "g4ion-binarycascade" "g4h-elastic_HP" "g4stopping"',
        "d:Ph/Default/CutForAllParticles = 0.05 mm",
    ])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output} for {table.n_material_sections} Schneider materials")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
