#!/usr/bin/env python3
"""Prepare portable TOPAS inputs for the MAIGO all-ion elastic package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SPECIES = [
    (1, 1), (1, 2), (1, 3), (2, 3), (2, 4), (2, 6),
    (3, 6), (3, 7), (4, 7), (4, 9), (4, 10), (5, 8),
    (5, 10), (5, 11), (6, 10), (6, 11), (6, 12), (4, 6),
]

TEMPLATE = """includeFile = {hu_file}
s:Ge/World/Material = \"Vacuum\"
d:Ge/World/HLX = 1 m
d:Ge/World/HLY = 1 m
d:Ge/World/HLZ = 1 m
b:Ge/World/Invisible = \"True\"
s:Ge/Patient/Parent = \"World\"
s:Ge/Patient/Material = \"G4_WATER\"
s:Ge/Patient/Type = \"TsDicomPatient\"
s:Ge/Patient/DicomDirectory = \"{dicom_dir}\"
b:Ge/Patient/PreLoadAllMaterials = \"True\"
sv:Ph/Default/Modules = 7 \"g4em-standard_opt4\" \"g4h-phy_QGSP_BIC_HP\" \"g4decay\" \"g4ion-inclxx\" \"g4h-elastic_HP\" \"g4stopping\" \"CarbonIonElasticPhysics\"
s:So/Demo/Type = \"Beam\"
s:So/Demo/Component = \"BeamPosition\"
s:So/Demo/BeamParticle = \"GenericIon({z},{a})\"
d:So/Demo/BeamEnergy = {beam_energy} MeV
u:So/Demo/BeamEnergySpread = 0
s:So/Demo/BeamPositionDistribution = \"None\"
s:So/Demo/BeamAngularDistribution = \"None\"
i:So/Demo/NumberOfHistoriesInRun = 1
i:Ts/NumberOfThreads = 1
i:Ts/Seed = {seed}
s:Sc/Elastic/Quantity = \"AllIonElasticDump\"
s:Sc/Elastic/Component = \"Patient\"
s:Sc/Elastic/OutputFile = \"{output_stem}\"
i:Sc/Elastic/ProjectileZ = {z}
i:Sc/Elastic/ProjectileA = {a}
i:Sc/Elastic/SamplesPerNode = {samples}
{recoil_flag}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hu-file", required=True, type=Path)
    parser.add_argument("--dicom-dir", required=True, type=Path,
                        help="A DICOM patient that instantiates all 25 Schneider sections")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=918721)
    parser.add_argument("--recoil-only", action="store_true")
    args = parser.parse_args()
    if not args.hu_file.is_file() or not args.dicom_dir.is_dir():
        parser.error("--hu-file and --dicom-dir must exist")
    if args.samples < 32:
        parser.error("--samples must be at least 32")
    args.output.mkdir(parents=True, exist_ok=True)
    species = [(1, 1)] if args.recoil_only else SPECIES
    cases = []
    for index, (z, a) in enumerate(species):
        stem = "recoil_stopping" if args.recoil_only else f"z{z}a{a}"
        text = TEMPLATE.format(
            hu_file=args.hu_file.resolve(), dicom_dir=args.dicom_dir.resolve(),
            z=z, a=a, beam_energy=200 * a, seed=args.seed + index,
            samples=args.samples, output_stem=(args.output / "raw" / stem).resolve(),
            recoil_flag='b:Sc/Elastic/RecoilStoppingOnly = "True"' if args.recoil_only else "",
        )
        path = args.output / f"{stem}.txt"
        path.write_text(text, encoding="utf-8")
        cases.append(path.name)
    (args.output / "cases.txt").write_text("\n".join(cases) + "\n", encoding="utf-8")
    (args.output / "campaign.json").write_text(json.dumps({
        "schema": 1, "species": species, "samples_per_node": args.samples,
        "hu_file": str(args.hu_file.resolve()), "dicom_dir": str(args.dicom_dir.resolve()),
        "recoil_only": args.recoil_only,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"prepared {len(cases)} case(s) below {args.output}")


if __name__ == "__main__":
    main()
