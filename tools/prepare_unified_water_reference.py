"""Prepare (never automatically submit) an independent 200 MeV/u water reference.

Finite scoring ROI in a wide water slab matches the native-water GPU ROI.
This is diagnostic, not authorization to remove the unvalidated-water gate.
"""
import argparse
import json
import yaml
from pathlib import Path
from run_topas10x_gpu_benchmark import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--gpu', type=Path, default=Path('/mnt/sda/wuwei/unified_water_50k_20260907'))
    parser.add_argument('--cpus', type=int, default=24)
    parser.add_argument('--seed', type=int, default=202609072)
    args = parser.parse_args()
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    topas = Path('/home/wuwei/topas/topas-build/topas')
    gpu = args.gpu.resolve()
    config = gpu / (gpu.name + '.yaml')
    cfg=yaml.safe_load(config.read_text())
    length=float(cfg['phantom_length_mm']); nz=int(cfg['voxel_bins_z'])
    energy=float(cfg['initial_energy_MeVu']); histories=int(cfg['number_of_histories'])
    if args.cpus<1 or args.cpus>96 or length!=nz*.5:
        raise ValueError('Unsupported resource/geometry request')
    if [cfg[k] for k in ('voxel_bins_x','voxel_bins_y','voxel_size_x_mm','voxel_size_y_mm','voxel_size_z_mm')]!=[64,64,2,2,.5]:
        raise ValueError('Only the frozen 2x2x0.5 mm ROI is supported')
    # TOPAS Beam direction is local +z; energy spread is PERCENT, GPU is fraction.
    text = f'''i:Ts/Seed = {args.seed}
i:Ts/NumberOfThreads = {args.cpus}
i:Ts/ShowHistoryCountAtInterval = 10000
b:Gr/Enable = "False"
s:Ge/World/Type = "TsBox"
s:Ge/World/Material = "Vacuum"
d:Ge/World/HLX = 2100 mm
d:Ge/World/HLY = 2100 mm
d:Ge/World/HLZ = {length+100} mm
s:Ge/Water/Parent = "World"
s:Ge/Water/Type = "TsBox"
s:Ge/Water/Material = "G4_WATER"
d:Ge/Water/HLX = 2000 mm
d:Ge/Water/HLY = 2000 mm
d:Ge/Water/HLZ = {length/2} mm
d:Ge/Water/TransZ = {length/2} mm
s:Ge/ROI/Parent = "Water"
s:Ge/ROI/Type = "TsBox"
s:Ge/ROI/Material = "G4_WATER"
d:Ge/ROI/HLX = 64 mm
d:Ge/ROI/HLY = 64 mm
d:Ge/ROI/HLZ = {length/2} mm
s:Ge/Source/Parent = "World"
s:Ge/Source/Type = "Group"
d:Ge/Source/TransZ = -0.001 mm
s:So/Beam/Type = "Beam"
s:So/Beam/Component = "Source"
s:So/Beam/BeamParticle = "GenericIon(6,12)"
d:So/Beam/BeamEnergy = {energy*12} MeV
u:So/Beam/BeamEnergySpread = {100*cfg['beam_energy_spread']}
s:So/Beam/BeamPositionDistribution = "None"
s:So/Beam/BeamAngularDistribution = "None"
i:So/Beam/NumberOfHistoriesInRun = {histories}
s:Ph/ListName = "Default"
s:Ph/Default/Type = "Geant4_Modular"
sv:Ph/Default/Modules = 7 "g4em-standard_opt4" "g4h-phy_QGSP_BIC_HP" "g4decay" "g4ion-inclxx" "g4h-elastic_HP" "g4stopping" "g4radioactivedecay"
s:Sc/Dose/Quantity = "DoseToMedium"
s:Sc/Dose/Component = "ROI"
i:Sc/Dose/XBins = 64
i:Sc/Dose/YBins = 64
i:Sc/Dose/ZBins = {nz}
s:Sc/Dose/OutputType = "Binary"
s:Sc/Dose/OutputFile = "{root}/dose_topas"
s:Sc/Dose/IfOutputFileAlreadyExists = "Exit"
'''
    em_only=not cfg['enable_inelastic']
    if em_only:
        text=text.replace('sv:Ph/Default/Modules = 7 "g4em-standard_opt4" "g4h-phy_QGSP_BIC_HP" "g4decay" "g4ion-inclxx" "g4h-elastic_HP" "g4stopping" "g4radioactivedecay"',
                          'sv:Ph/Default/Modules = 2 "g4em-standard_opt4" "g4decay"')
    (root / 'topas.txt').write_text(text)
    (root / 'run.slurm').write_text(f'''#!/bin/bash
#SBATCH --job-name=water_unified_ref
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task={args.cpus}
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output={root}/job_%j.log
#SBATCH --error={root}/job_%j.err
set -euo pipefail
cd {root}
{topas} {root}/topas.txt
''')
    files = [topas, config, gpu/'manifest.json', gpu/'dose.raw',
             root/'topas.txt', root/'run.slurm', Path(__file__).resolve()]
    manifest = dict(status='INDEPENDENT_REFERENCE_PENDING_NOT_VALIDATED',
                    pins={str(p): sha(p) for p in files}, histories=histories,
                    em_only=em_only,seed=args.seed,energy_MeVu=energy, nz=nz, length_mm=length,
                    gpu_directory=str(gpu), cpus=args.cpus, memory_GiB=24,
                    limitations=['Finite 4m-wide slab approximates laterally infinite GPU water.',
                                 'TOPAS full physics versus capped GPU cascade, no explicit water electrons.',
                                 'Production cuts are TOPAS defaults; no custom cut or fitted normalization.'])
    (root/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(root/'run.slurm')


if __name__ == '__main__':
    main()
