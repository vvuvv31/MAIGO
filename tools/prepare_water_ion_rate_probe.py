"""Read-only Geant4 G4_WATER rate probe, not a production table generation."""
import argparse,json
from pathlib import Path
from run_topas10x_gpu_benchmark import sha

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    root=p.parse_args().out.resolve();root.mkdir(parents=True,exist_ok=False)
    source=Path('/mnt/sda/wuwei/unified_water_ref300_seed2_20260907/topas.txt')
    text=source.read_text().replace(str(source.parent),str(root))
    text=text.replace('NumberOfThreads = 72','NumberOfThreads = 1').replace('NumberOfHistoriesInRun = 50000','NumberOfHistoriesInRun = 1')
    text+=f'''s:Sc/Rates/Quantity = "IonCrossSectionNtuple"
s:Sc/Rates/Component = "ROI"
s:Sc/Rates/OutputType = "ASCII"
s:Sc/Rates/OutputFile = "{root}/rates"
s:Sc/Rates/IfOutputFileAlreadyExists = "Exit"
'''
    (root/'topas.txt').write_text(text);exe=Path('/home/wuwei/topas/topas-build/topas')
    (root/'run.slurm').write_text(f'''#!/bin/bash
#SBATCH --job-name=water_ion_rates
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
    files=[source,exe,root/'topas.txt',root/'run.slurm',Path(__file__).resolve(),
        Path('/home/wuwei/topas/extensions/IonCrossSectionNtuple.cc')]
    files += [Path('data/schneider/secondary_inelastic_rates_v2_1.bin').resolve(),Path('data/water_unified/g4_water_material.json').resolve()]
    (root/'manifest.json').write_text(json.dumps(dict(status='READ_ONLY_RATE_DIAGNOSTIC',
        pins={str(f):sha(f) for f in files},limitations=['0.01 to 400.01 MeV/u, 1 MeV/u spacing.',
        'Runtime uses final-state domain masking; compare fully covered energies separately.',
        '3D dose is incidental one-history output, not a dose comparison.']),indent=2))

if __name__=='__main__':main()
