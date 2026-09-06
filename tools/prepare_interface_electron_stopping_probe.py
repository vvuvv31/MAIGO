"""Two-material intrinsic electron stopping diagnostic using an EXISTING scorer.

No TOPAS source modification or dose fit. Local Slurm only, 2 CPUs / 8 GiB total.
Keep the source geometry/material definitions and 3D DoseToMedium scorer.
"""
import argparse,json,re
from pathlib import Path
from analyze_longitudinal_holdout import sha

def prepare(source,out):
    manifest=json.loads((source/'manifest.json').read_text());out.mkdir(exist_ok=False)
    report=dict(status='INTRINSIC_STOPPING_DIAGNOSTIC_ONLY',max_cpus=2,max_memory_GiB=8,cases={})
    for material in ('air','tissue'):
        src=source/material/'9181861';dest=out/material;dest.mkdir()
        if sha(src/'run.txt')!=manifest['inputs'][str(src/'run.txt')]:raise ValueError('Source configuration pin')
        text=(src/'run.txt').read_text().replace(str(src),str(dest))
        text=re.sub(r'NumberOfHistoriesInRun = 4','NumberOfHistoriesInRun = 1',text)
        text=re.sub(r'Ts/NumberOfThreads = 4','Ts/NumberOfThreads = 1',text)
        text=re.sub(r'^.*Sc/ElectronDeposit/.*\n','',text,flags=re.M)
        text+='\ns:Sc/ElectronTransport/Quantity = "ElectronTransportNtuple"\ns:Sc/ElectronTransport/Component = "ZResponse"\nb:Sc/ElectronTransport/PropagateToChildren = "False"\ns:Sc/ElectronTransport/OutputType = "ASCII"\n'
        text+=f's:Sc/ElectronTransport/OutputFile = "{dest}/electron_stopping"\n'
        (dest/'run.txt').write_text(text)
        (dest/'run.slurm').write_text(f'#!/bin/bash\n#SBATCH --job-name=electron_sp_{material}\n#SBATCH --partition=compute\n#SBATCH --nodes=1\n#SBATCH --cpus-per-task=1\n#SBATCH --mem=4G\n#SBATCH --time=00:10:00\n#SBATCH --output={dest}/job_%j.log\n#SBATCH --error={dest}/job_%j.err\nset -euo pipefail\ncd {dest}\ntest ! -e electron_stopping.phsp\n/home/wuwei/topas/topas-build/topas {dest}/run.txt\n')
        report['cases'][material]=dict(source_config_sha256=sha(src/'run.txt'))
    report['inputs']={str(p):sha(p) for p in out.rglob('*') if p.is_file()}
    for p in (Path('/home/wuwei/topas/topas-build/topas'),Path('/home/wuwei/topas/extensions/ElectronTransportNtuple.cc')):report['inputs'][str(p)]=sha(p)
    with (out/'manifest.json').open('x') as f:json.dump(report,f,indent=2)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();prepare(a.source.resolve(),a.out.resolve())
