"""Measure actual upstream-air C12/electron phase space; no dose fit."""
import re
from pathlib import Path

def main():
    old=Path('/mnt/sda/wuwei/ct_ideal_pencil_20260906/em')
    root=Path('/mnt/sda/wuwei/ct_entrance_phase_r2_20260906');root.mkdir(exist_ok=False)
    text=(old/'topas.txt').read_text()
    text='\n'.join(line for line in text.splitlines() if 'Sc/OSMK_Dtotal/' not in line)+'\n'
    text=text.replace('NumberOfThreads = 48','NumberOfThreads = 24')
    # Original patient low Y=-151.75+9.3887 mm. Surface is 0.01 mm upstream.
    text+='''
s:Ge/Entrance/Type = "TsBox"
s:Ge/Entrance/Parent = "World"
s:Ge/Entrance/Material = "Air"
d:Ge/Entrance/HLX = 300 mm
d:Ge/Entrance/HLY = 100 mm
d:Ge/Entrance/HLZ = 0.001 mm
d:Ge/Entrance/TransY = -142.3723 mm
d:Ge/Entrance/RotX = 90 deg
s:Sc/Entrance/Quantity = "PhaseSpace"
s:Sc/Entrance/Surface = "Entrance/ZPlusSurface"
s:Sc/Entrance/OnlyIncludeParticlesGoing = "Out"
s:Sc/Entrance/OutputType = "ASCII"
b:Sc/Entrance/KillAfterPhaseSpace = "True"
b:Sc/Entrance/IncludeParentID = "True"
s:Sc/Entrance/IfOutputFileAlreadyExists = "Exit"
'''
    text+=f's:Sc/Entrance/OutputFile = "{root}/entrance"\n'
    (root/'run.txt').write_text(text)
    (root/'run.slurm').write_text(f'#!/bin/bash\n#SBATCH --job-name=ct_entrance_phase\n#SBATCH --partition=compute\n#SBATCH --nodes=1\n#SBATCH --cpus-per-task=24\n#SBATCH --mem=16G\n#SBATCH --time=00:30:00\n#SBATCH --output={root}/job_%j.log\n#SBATCH --error={root}/job_%j.err\nset -euo pipefail\ncd {root}\n/home/wuwei/topas/topas-build/topas {root}/run.txt\n')

if __name__=='__main__':main()
