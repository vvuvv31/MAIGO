"""Prepare independent birth/no-scorer controls; never overwrite frozen runs."""
import argparse,json
from pathlib import Path
from run_topas10x_gpu_benchmark import sha

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mode',choices=['birth','control','original'],required=True)
    a=p.parse_args();root=a.out.resolve();root.mkdir(parents=True,exist_ok=False)
    source=Path('/mnt/sda/wuwei/unified_water_ref300_seed2_20260907')
    text=(source/'topas.txt').read_text().replace(str(source),str(root))
    if a.mode!='control':
        quantity='CarbonBirthOnlyNtuple' if a.mode=='birth' else 'CarbonCascadeNtuple'
        text+=f'''s:Sc/Cascade/Quantity = "{quantity}"
s:Sc/Cascade/Component = "Water"
b:Sc/Cascade/PropagateToChildren = "True"
s:Sc/Cascade/OutputType = "ASCII"
s:Sc/Cascade/OutputFile = "{root}/cascade"
s:Sc/Cascade/IfOutputFileAlreadyExists = "Exit"
'''
    (root/'topas.txt').write_text(text)
    exe=Path('/home/wuwei/topas/topas-build/topas')
    (root/'run.slurm').write_text(f'''#!/bin/bash
#SBATCH --job-name=water_birth_{a.mode}
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
        Path('/mnt/sda/wuwei/unified_water300_origin_20260907/energy_ledger.json')]
    for name in ['CarbonCascadeNtuple','CarbonBirthOnlyNtuple']:
        for ext in ['cc','hh']:files.append(Path('/home/wuwei/topas/extensions')/(name+'.'+ext))
    limitations=['First-tracked products, not complete vertex production.',
        'First-step nuclear interactions and parent context overwrites can omit or misassociate births.',
        'GPU queued_birth minus reaction_import excludes unqueued cutoff products.',
        'Column 17 is unmeasured zero for birth-only scorer; not a physical cross section.',
        'Cascade births are not independent beam energy.']
    (root/'manifest.json').write_text(json.dumps(dict(status='DIAGNOSTIC_PENDING',mode=a.mode,
        histories=50000,baseline=str(source/'dose_topas.bin'),
        pins={str(f):sha(f) for f in files},limitations=limitations),indent=2))

if __name__=='__main__':main()
