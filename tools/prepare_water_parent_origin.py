"""Reuse existing parent-origin scorer with independent neutral ancestry veto."""
import argparse,json
from pathlib import Path
from run_topas10x_gpu_benchmark import sha

MAPPING={'primary_c':'primary','secondary_c':'secondary_carbon','boron':'secondary_boron',
         'beryllium':'secondary_beryllium','lithium':'secondary_lithium',
         'helium':'secondary_helium','z1':'secondary_proton','other_charged':'secondary_other_charged',
         'unclassified':'unclassified'}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    root=p.parse_args().out.resolve();root.mkdir(parents=True,exist_ok=False)
    source=Path('/mnt/sda/wuwei/unified_water_ref300_seed2_20260907')
    text=(source/'topas.txt').read_text().split('s:Sc/Dose/Quantity')[0]
    text=text.replace('i:Ts/NumberOfThreads = 72','i:Ts/NumberOfThreads = 64')
    for name in ['total','neutral_lineage',*MAPPING]:
        quantity='ChargedOriginDoseToMedium' if name in MAPPING else 'DoseToMedium'
        text+=f'''s:Sc/{name}/Quantity = "{quantity}"
s:Sc/{name}/Component = "ROI"
i:Sc/{name}/XBins = 64
i:Sc/{name}/YBins = 64
i:Sc/{name}/ZBins = 800
s:Sc/{name}/OutputType = "Binary"
s:Sc/{name}/OutputFile = "{root}/{name}"
s:Sc/{name}/IfOutputFileAlreadyExists = "Exit"
'''
        if name=='neutral_lineage':
            text+='sv:Sc/neutral_lineage/OnlyIncludeIfParticleOrAncestorNamed = 2 "neutron" "gamma"\n'
        if name in MAPPING:
            text+=f's:Sc/{name}/OriginCategory = "{name}"\n'
            text+=f'sv:Sc/{name}/OnlyIncludeIfParticleOrAncestorNotNamed = 2 "neutron" "gamma"\n'
    (root/'topas.txt').write_text(text)
    exe=Path('/home/wuwei/topas/topas-build/topas')
    (root/'run.slurm').write_text(f'''#!/bin/bash
#SBATCH --job-name=water_parent_origin
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output={root}/job_%j.log
#SBATCH --error={root}/job_%j.err
set -euo pipefail
cd {root}
{exe} {root}/topas.txt
''')
    files=[source/'topas.txt',source/'dose_topas.bin',exe,root/'topas.txt',root/'run.slurm',Path(__file__).resolve()]
    files += [Path('/home/wuwei/topas/extensions')/f'ChargedOriginDoseToMedium.{ext}' for ext in ['cc','hh']]
    (root/'manifest.json').write_text(json.dumps(dict(status='DIAGNOSTIC_PENDING',mapping=MAPPING,
        baseline=str(source/'dose_topas.bin'),pins={str(f):sha(f) for f in files},
        limitations=['PrimaryC uses ParentID0, not first-nuclear-interaction status.',
                     'Parents absent from ROI can yield unclassified electronic dose.',
                     'Electrons are scored at their deposit position, not moved back to their birthplace.',
                     'Neutron/gamma ancestry veto is independent of scorer parent cache.']),indent=2))

if __name__=='__main__':main()
