#!/usr/bin/env python3
"""Prepare water/air/copper stopping, copper elastic/rate and material probe jobs."""
import argparse
import json
from pathlib import Path
import subprocess


def card(material, particle, energy, quantities, samples):
    text = f'''i:Ts/NumberOfThreads = 1
i:Ts/Seed = 92617000
b:Ts/PauseBeforeQuit = "False"
b:Gr/Enable = "False"
s:Ge/World/Material = "Vacuum"
d:Ge/World/HLX = 1000 mm
d:Ge/World/HLY = 1000 mm
d:Ge/World/HLZ = 1000 mm
s:Ge/Target/Parent = "World"
s:Ge/Target/Type = "TsBox"
s:Ge/Target/Material = "{material}"
d:Ge/Target/HLX = 100 mm
d:Ge/Target/HLY = 100 mm
d:Ge/Target/HLZ = 100 mm
s:Ge/BeamPosition/Parent = "World"
s:Ge/BeamPosition/Type = "Group"
s:So/Beam/Type = "Beam"
s:So/Beam/Component = "BeamPosition"
s:So/Beam/BeamParticle = "{particle}"
d:So/Beam/BeamEnergy = {energy} MeV
u:So/Beam/BeamEnergySpread = 0
s:So/Beam/BeamPositionDistribution = "None"
s:So/Beam/BeamAngularDistribution = "None"
i:So/Beam/NumberOfHistoriesInRun = 1
sv:Ph/Default/Modules = 7 "g4em-standard_opt4" "g4h-phy_QGSP_BIC_HP" "g4decay" "g4ion-inclxx" "g4h-elastic_HP" "g4stopping" "CarbonIonElasticPhysics"
d:Ph/Default/CutForAllParticles = 0.05 mm
d:Ph/Default/EMRangeMax = 10 GeV
'''
    for name, quantity in quantities:
        text += f'''s:Sc/{name}/Quantity = "{quantity}"
s:Sc/{name}/Component = "Target"
s:Sc/{name}/OutputType = "ASCII"
s:Sc/{name}/OutputFile = "{name}"
s:Sc/{name}/IfOutputFileAlreadyExists = "Exit"
'''
        if quantity == 'PrimaryMaterialDump':
            text += f'd:Sc/{name}/MaximumEnergyPerNucleon = 400 MeV\ni:Sc/{name}/SamplesPerNode = {samples}\n'
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--topas', type=Path, help='Run the prepared jobs locally')
    parser.add_argument('--samples-per-node', type=int, default=512)
    args = parser.parse_args()
    if not 1 <= args.samples_per_node <= 1000000:
        parser.error('Samples per node outside exporter range')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    tasks = []
    for medium, material in [('water', 'Water_75eV'), ('air', 'G4_AIR'), ('copper', 'G4_Cu')]:
        for species, particle, energy in [('proton', 'proton', 200), ('carbon', 'GenericIon(6,12,6)', 2400)]:
            quantities = [('primary', 'PrimaryMaterialDump' if medium == 'copper' else 'CarbonStoppingPowerNtuple')]
            if medium in ('water', 'copper'):
                quantities.append(('ions', 'IonStoppingPowerNtuple'))
            if medium == 'copper':
                quantities.append(('ion_rates', 'IonCrossSectionNtuple'))
            case = out/f'{medium}_{species}'
            case.mkdir()
            (case/'topas.txt').write_text(card(material, particle, energy, quantities, args.samples_per_node))
            tasks.append(dict(case=case.name, material=material, species=species))
    case = out/'water_material'; case.mkdir()
    (case/'topas.txt').write_text(card('G4_WATER', 'proton', 200, [('material', 'MaterialDefinitionDump')], args.samples_per_node))
    tasks.append(dict(case=case.name, material='G4_WATER'))
    (out/'campaign.json').write_text(json.dumps(dict(tasks=tasks), indent=2)+'\n')
    if args.topas:
        for task in tasks:
            case = out/task['case']
            with (case/'topas.log').open('x') as stream:
                subprocess.run([str(args.topas.resolve()), 'topas.txt'], cwd=case,
                               stdout=stream, stderr=subprocess.STDOUT, check=True)
            print(task['case'], 'completed', flush=True)


if __name__ == '__main__':
    main()
