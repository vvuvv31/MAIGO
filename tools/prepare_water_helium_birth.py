"""Read-only first-tracked cascade products plus unchanged 3D reference dose."""
import argparse,json
from pathlib import Path
from run_topas10x_gpu_benchmark import sha

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    root=p.parse_args().out.resolve();root.mkdir(parents=True,exist_ok=False)
    source=Path('/mnt/sda/wuwei/unified_water_ref300_seed2_20260907')
    text=(source/'topas.txt').read_text().replace(str(source),str(root))
    text+=f'''s:Sc/Cascade/Quantity = "CarbonCascadeNtuple"
s:Sc/Cascade/Component = "Water"
b:Sc/Cascade/PropagateToChildren = "True"
s:Sc/Cascade/OutputType = "ASCII"
s:Sc/Cascade/OutputFile = "{root}/cascade"
s:Sc/Cascade/IfOutputFileAlreadyExists = "Exit"
'''
    (root/'topas.txt').write_text(text);exe=Path('/home/wuwei/topas/topas-build/topas')
    (root/'run.slurm').write_text(f'''#!/bin/bash
#SBATCH --job-name=water_helium_birth
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output={root}/job_%j.log
#SBATCH --error={root}/job_%j.err
set -euo pipefail
cd {root}
{exe} {root}/topas.txt
''')
    files=[source/'topas.txt',source/'dose_topas.bin',root/'topas.txt',root/'run.slurm',exe,Path(__file__).resolve(),
           Path('/home/wuwei/topas/extensions/CarbonCascadeNtuple.cc'),Path('/home/wuwei/topas/extensions/CarbonCascadeNtuple.hh'),
           Path('/mnt/sda/wuwei/unified_water300_origin_20260907/energy_ledger.json')]
    (root/'manifest.json').write_text(json.dumps(dict(status='DIAGNOSTIC_PENDING',histories=50000,
        baseline=str(source/'dose_topas.bin'),pins={str(f):sha(f) for f in files},
        limitations=['Product rows are first-tracked products, not complete vertex production.',
                     'A product undergoing nuclear interaction on its first step may lack a separate product row.',
                     'GPU queued_birth minus reaction_import isolates primary-queued energy, not all products below cutoff.',
                     'Do not equate sums over all cascade births to independent beam energy.']),indent=2))

if __name__=='__main__':main()
