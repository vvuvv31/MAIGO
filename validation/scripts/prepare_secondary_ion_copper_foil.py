#!/usr/bin/env python3
"""Prepare independent TOPAS secondary-ion transport benchmarks in Copper."""

from __future__ import annotations

import argparse
from pathlib import Path


SPECIES = {
    "he3": (2, 3),
    "he4": (2, 4),
    "li7": (3, 7),
    "be9": (4, 9),
    "b11": (5, 11),
}


def case_text(
    label: str,
    atomic_number: int,
    mass_number: int,
    energy_mevu: int,
    thickness_mm: float,
    histories: int,
    em_range_max_mev: float,
) -> str:
    total_energy_mev = mass_number * energy_mevu
    half_thickness = 0.5 * thickness_mm
    source_z = -half_thickness - 10.0
    entrance_z = -half_thickness - 0.015
    exit_z = half_thickness + 0.015
    thickness_tag = f"{thickness_mm:g}".replace(".", "p")
    em_range_tag = f"{em_range_max_mev:g}".replace(".", "p")
    stem = (
        f"secondary_ion_copper_{label}_e{energy_mevu}_"
        f"t{thickness_tag}_{histories}_emmax{em_range_tag}"
    )
    return f"""# Generated independent secondary-ion Copper foil benchmark.
i:Ts/Seed = {20260801 + atomic_number * 100 + mass_number * 10 + energy_mevu}
i:Ts/NumberOfThreads = 40
i:Ts/ShowHistoryCountAtInterval = {max(1000, histories // 10)}
b:Gr/Enable = "False"
b:Ge/CheckForOverlaps = "True"

s:Ge/World/Type = "TsBox"
s:Ge/World/Material = "G4_Galactic"
d:Ge/World/HLX = 100 mm
d:Ge/World/HLY = 100 mm
d:Ge/World/HLZ = 100 mm

s:Ge/CopperTarget/Type = "TsBox"
s:Ge/CopperTarget/Parent = "World"
s:Ge/CopperTarget/Material = "Copper"
d:Ge/CopperTarget/HLX = 50 mm
d:Ge/CopperTarget/HLY = 50 mm
d:Ge/CopperTarget/HLZ = {half_thickness:g} mm
d:Ge/CopperTarget/MaxStepSize = 0.05 mm

s:Ge/EntranceFilm/Type = "TsBox"
s:Ge/EntranceFilm/Parent = "World"
s:Ge/EntranceFilm/Material = "G4_Galactic"
d:Ge/EntranceFilm/HLX = 50 mm
d:Ge/EntranceFilm/HLY = 50 mm
d:Ge/EntranceFilm/HLZ = 0.005 mm
d:Ge/EntranceFilm/TransZ = {entrance_z:g} mm

s:Ge/ExitFilm/Type = "TsBox"
s:Ge/ExitFilm/Parent = "World"
s:Ge/ExitFilm/Material = "G4_Galactic"
d:Ge/ExitFilm/HLX = 50 mm
d:Ge/ExitFilm/HLY = 50 mm
d:Ge/ExitFilm/HLZ = 0.005 mm
d:Ge/ExitFilm/TransZ = {exit_z:g} mm

s:Ge/BeamPosition/Type = "Group"
s:Ge/BeamPosition/Parent = "World"
d:Ge/BeamPosition/TransZ = {source_z:g} mm
d:Ge/BeamPosition/RotY = 180 deg

s:So/IonBeam/Type = "Beam"
s:So/IonBeam/Component = "BeamPosition"
s:So/IonBeam/BeamParticle = "GenericIon({atomic_number},{mass_number},{atomic_number})"
d:So/IonBeam/BeamEnergy = {total_energy_mev} MeV
u:So/IonBeam/BeamEnergySpread = 0
s:So/IonBeam/BeamPositionDistribution = "None"
s:So/IonBeam/BeamAngularDistribution = "None"
i:So/IonBeam/NumberOfHistoriesInRun = {histories}

sv:Ph/Default/Modules = 6 "g4em-standard_opt4" "g4h-phy_QGSP_BERT_HP" "g4decay" "g4ion-binarycascade" "g4h-elastic_HP" "g4stopping"
d:Ph/Default/CutForAllParticles = 0.05 mm
d:Ph/Default/EMRangeMax = {em_range_max_mev:g} MeV

s:Sc/IonReactions/Quantity = "CarbonReactionNtuple"
s:Sc/IonReactions/Component = "CopperTarget"
s:Sc/IonReactions/OutputFile = "output/{stem}_reactions"
s:Sc/IonReactions/OutputType = "ASCII"
s:Sc/IonReactions/IfOutputFileAlreadyExists = "Overwrite"

s:Sc/EntrancePhaseSpace/Quantity = "PhaseSpace"
s:Sc/EntrancePhaseSpace/Surface = "EntranceFilm/ZMinusSurface"
s:Sc/EntrancePhaseSpace/OnlyIncludeParticlesGoing = "In"
s:Sc/EntrancePhaseSpace/OutputType = "ASCII"
s:Sc/EntrancePhaseSpace/OutputFile = "output/{stem}_entrance"
s:Sc/EntrancePhaseSpace/IfOutputFileAlreadyExists = "Overwrite"
i:Sc/EntrancePhaseSpace/OutputBufferSize = 100000
b:Sc/EntrancePhaseSpace/IncludeRunID = "True"
b:Sc/EntrancePhaseSpace/IncludeEventID = "True"
b:Sc/EntrancePhaseSpace/IncludeTrackID = "True"
b:Sc/EntrancePhaseSpace/IncludeParentID = "True"

s:Sc/ExitPhaseSpace/Quantity = "PhaseSpace"
s:Sc/ExitPhaseSpace/Surface = "ExitFilm/ZMinusSurface"
s:Sc/ExitPhaseSpace/OnlyIncludeParticlesGoing = "In"
s:Sc/ExitPhaseSpace/OutputType = "ASCII"
s:Sc/ExitPhaseSpace/OutputFile = "output/{stem}_exit"
s:Sc/ExitPhaseSpace/IfOutputFileAlreadyExists = "Overwrite"
i:Sc/ExitPhaseSpace/OutputBufferSize = 100000
b:Sc/ExitPhaseSpace/IncludeRunID = "True"
b:Sc/ExitPhaseSpace/IncludeEventID = "True"
b:Sc/ExitPhaseSpace/IncludeTrackID = "True"
b:Sc/ExitPhaseSpace/IncludeParentID = "True"
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--species",
        nargs="+",
        choices=tuple(SPECIES),
        default=["he3", "he4"],
    )
    parser.add_argument("--energies", type=int, nargs="+", default=[300])
    parser.add_argument(
        "--thicknesses-mm", type=float, nargs="+", default=[30.0, 60.0]
    )
    parser.add_argument("--histories", type=int, default=50_000)
    parser.add_argument(
        "--em-range-max-mev",
        type=float,
        default=6000.0,
        help="Upper energy of the Geant4 EM tables (total kinetic energy).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("ct/minibeam")
    )
    args = parser.parse_args()
    if (
        args.histories <= 0
        or args.em_range_max_mev <= 0.0
        or any(energy <= 0 for energy in args.energies)
        or any(thickness <= 0.0 for thickness in args.thicknesses_mm)
    ):
        raise ValueError("histories, energies, and thicknesses must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for label in args.species:
        atomic_number, mass_number = SPECIES[label]
        for energy in args.energies:
            for thickness in args.thicknesses_mm:
                thickness_tag = f"{thickness:g}".replace(".", "p")
                output = args.output_dir / (
                    f"run_secondary_ion_copper_{label}_e{energy}_"
                    f"t{thickness_tag}_{args.histories}_"
                    f"emmax{f'{args.em_range_max_mev:g}'.replace('.', 'p')}.txt"
                )
                output.write_text(
                    case_text(
                        label,
                        atomic_number,
                        mass_number,
                        energy,
                        thickness,
                        args.histories,
                        args.em_range_max_mev,
                    ),
                    encoding="utf-8",
                )
                print(output)


if __name__ == "__main__":
    main()
