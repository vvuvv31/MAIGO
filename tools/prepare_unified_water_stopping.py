"""Prepare matched 78 eV G4_WATER stopping extraction; do not overwrite old data."""
import argparse
import json
import re
from pathlib import Path
from run_topas10x_gpu_benchmark import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); root=a.out.resolve()
    source=Path('/mnt/sda/wuwei/unified_water_topas_200_20260907/topas.txt')
    text=source.read_text()
    text=re.sub(r'i:Ts/NumberOfThreads = 24', 'i:Ts/NumberOfThreads = 1',text)
    text=text.replace('NumberOfHistoriesInRun = 50000','NumberOfHistoriesInRun = 1')
    text=text[:text.index('s:Sc/Dose/Quantity')]
    for label,quantity in [('carbon','CarbonStoppingPowerNtuple'),('ions','IonStoppingPowerNtuple')]:
        text+=f'''s:Sc/{label}/Quantity = "{quantity}"
s:Sc/{label}/Component = "ROI"
s:Sc/{label}/OutputType = "ASCII"
s:Sc/{label}/OutputFile = "{root}/{label}"
s:Sc/{label}/IfOutputFileAlreadyExists = "Exit"
'''
    root.mkdir(parents=True,exist_ok=False)
    (root/'topas.txt').write_text(text)
    exe=Path('/home/wuwei/topas/topas-build/topas')
    (root/'run.slurm').write_text(f'''#!/bin/bash
#SBATCH --job-name=water78_stopping
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output={root}/job_%j.log
#SBATCH --error={root}/job_%j.err
set -euo pipefail
cd {root}
{exe} {root}/topas.txt
''')
    files=[source,exe,root/'topas.txt',root/'run.slurm',Path(__file__).resolve()]
    files += [Path('/home/wuwei/topas/extensions')/(name+'.cc') for name in
              ('CarbonStoppingPowerNtuple','IonStoppingPowerNtuple')]
    (root/'manifest.json').write_text(json.dumps(dict(status='CANDIDATE_NOT_VALIDATED',
        material='G4_WATER',mean_excitation_energy_eV=78,
        pins={str(f):sha(f) for f in files},
        limitations=['Existing extractors cover 0.01–400.01 MeV/u only.',
                     'Ion restricted column uses 0.05 mm cut; electronic column is unrestricted.']),indent=2))
    print(root/'run.slurm')


if __name__=='__main__':main()
