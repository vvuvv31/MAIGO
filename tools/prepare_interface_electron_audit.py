"""Bounded v3 observer runs on the validated mass-voxel interface geometry."""
import argparse,json,re
from pathlib import Path
from analyze_longitudinal_holdout import sha

def prepare(reference,out):
    out.mkdir(parents=True,exist_ok=False)
    manifest=dict(scope="observer only; no response fit",histories_per_job=4,
                  max_cpus=8,max_mem_GiB=32,file_limit_GiB=1,cases={})
    for name in ("air_to_tissue","tissue_to_air"):
        for seed in ("s1","s2"):
            source=reference/name/seed
            dest=out/name/seed;dest.mkdir(parents=True)
            text=(source/"run.txt").read_text().replace(str(source),str(dest))
            text=re.sub(r'NumberOfHistoriesInRun = 20000', 'NumberOfHistoriesInRun = 4',text)
            text=text.replace('Ts/NumberOfThreads = 24','Ts/NumberOfThreads = 2')
            text+='\ns:Sc/ElectronDeposit/Quantity = "CarbonElectronDepositNtupleV3"\ns:Sc/ElectronDeposit/Component = "ZPhantom"\nb:Sc/ElectronDeposit/PropagateToChildren = "True"\ns:Sc/ElectronDeposit/OutputType = "Binary"\n'
            text+=f's:Sc/ElectronDeposit/OutputFile = "{dest}/steps"\ns:Sc/ElectronDeposit/IfOutputFileAlreadyExists = "Overwrite"\n'
            (dest/"run.txt").write_text(text)
            (dest/"run.slurm").write_text(f'#!/bin/bash\n#SBATCH --job-name=iface_el_{name}_{seed}\n#SBATCH --partition=compute\n#SBATCH --nodes=1\n#SBATCH --cpus-per-task=2\n#SBATCH --mem=8G\n#SBATCH --time=00:15:00\n#SBATCH --output={dest}/job_%j.log\n#SBATCH --error={dest}/job_%j.err\nset -euo pipefail\n# 1 GiB per file, bounded histories; never truncate or reuse a result.\nulimit -f 1048576\ncd {dest}\ntest ! -e steps.phsp\n/home/wuwei/topas/topas-build/topas {dest}/run.txt\n')
            manifest['cases'][name+'/'+seed]=dict(reference_config_sha256=sha(source/'run.txt'))
    manifest['inputs']={str(p):sha(p) for p in out.rglob('*') if p.is_file()}
    for f in ['/home/wuwei/topas/topas-build/topas','/home/wuwei/topas/extensions/CarbonElectronDepositNtupleV3.cc','/home/wuwei/topas/extensions/CarbonElectronDepositNtupleV3.hh']:
        manifest['inputs'][f]=sha(Path(f))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('reference',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();prepare(a.reference.resolve(),a.out.resolve())
