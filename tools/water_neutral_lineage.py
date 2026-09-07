"""Local TOPAS 3D lineage partition; diagnostic only, no physics changes."""
import argparse,json,re,subprocess
from pathlib import Path
import numpy as np
from run_topas10x_gpu_benchmark import sha

GROUPS={
    'total':[],
    'neutron':['sv:Sc/{name}/OnlyIncludeIfParticleOrAncestorNamed = 1 "neutron"'],
    'gamma_no_neutron':['sv:Sc/{name}/OnlyIncludeIfParticleOrAncestorNamed = 1 "gamma"',
                        'sv:Sc/{name}/OnlyIncludeIfParticleOrAncestorNotNamed = 1 "neutron"'],
    'neither':['sv:Sc/{name}/OnlyIncludeIfParticleOrAncestorNotNamed = 2 "neutron" "gamma"']}

def prepare(root):
    source=Path('/mnt/sda/wuwei/unified_water_ref300_seed2_20260907')
    text=(source/'topas.txt').read_text()
    text=text[:text.index('s:Sc/Dose/Quantity')]
    for name,filters in GROUPS.items():
        text+=f'''s:Sc/{name}/Quantity = "DoseToMedium"
s:Sc/{name}/Component = "ROI"
i:Sc/{name}/XBins = 64
i:Sc/{name}/YBins = 64
i:Sc/{name}/ZBins = 800
s:Sc/{name}/OutputType = "Binary"
s:Sc/{name}/OutputFile = "{root}/{name}"
s:Sc/{name}/IfOutputFileAlreadyExists = "Exit"
'''
        text+='\n'.join(line.format(name=name) for line in filters)+'\n'
    root.mkdir(parents=True,exist_ok=False)
    (root/'topas.txt').write_text(text)
    exe=Path('/home/wuwei/topas/topas-build/topas')
    (root/'run.slurm').write_text(f'''#!/bin/bash
#SBATCH --job-name=water_lineage
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
    files=[source/'topas.txt',source/'dose_topas.bin',exe,root/'topas.txt',root/'run.slurm',Path(__file__).resolve(),
           Path('/home/wuwei/topas/OpenTOPAS-4.2.3/filtering/TsFilterByType.cc')]
    (root/'manifest.json').write_text(json.dumps(dict(status='LINEAGE_DIAGNOSTIC_PENDING',histories=50000,
        pins={str(f):sha(f) for f in files},baseline=str(source/'dose_topas.bin'),
        partition='neutron ancestry; gamma ancestry excluding neutron; neither ancestry. All include current particle.',
        limitations=['Neither is NOT a primary-only or charged-only scorer.',
                     'Gamma ancestry includes EM photons; not all gamma ancestry is missing on GPU.',
                     'A lineage partition is not a causal switch-off experiment.']),indent=2))

def analyze(root,job):
    status=subprocess.check_output(['sacct','-j',str(job),'--format=JobIDRaw,State,ExitCode','-n','-P'],text=True)
    if f'{job}|COMPLETED|0:0' not in status.splitlines():raise ValueError('Job not completed: '+status)
    log=(root/f'job_{job}.log').read_text()
    if 'Finalization:' not in log or 'Total number of histories: 50000' not in log:raise ValueError('Incomplete run')
    m=json.loads((root/'manifest.json').read_text())
    for name,h in m['pins'].items():
        if sha(Path(name))!=h:raise ValueError('Frozen input changed: '+name)
    data={}
    for name in GROUPS:
        header=(root/f'{name}.binheader').read_text()
        if [int(re.search(rf'# {axis} in (\d+) bins',header)[1]) for axis in 'XYZ']!=[64,64,800]:raise ValueError('Grid mismatch')
        if 'DoseToMedium ( Gy ) : Sum' not in header:raise ValueError('Wrong units/column')
        arr=np.fromfile(root/f'{name}.bin','<f8').reshape(800,64,64)
        if not np.isfinite(arr).all() or arr.min()<0:raise ValueError('Invalid payload')
        data[name]=arr
    total=data['total'];rest=sum(data[k] for k in GROUPS if k!='total')
    if total.max()<=0:raise ValueError('Empty total')
    closure=float(np.max(np.abs(rest-total))/total.max())
    if closure>1e-10:raise ValueError('Lineage partition does not close: '+str(closure))
    baseline=np.fromfile(m['baseline'],'<f8').reshape(total.shape)
    delta=float(np.max(np.abs(total-baseline))/baseline.max())
    factor=2e-6*6.241509074e12/50000
    energies={k:float(v.sum()*factor) for k,v in data.items()}
    gpu=Path('/mnt/sda/wuwei/unified_water300_seed2_20260907/dose.raw')
    g=np.fromfile(gpu,'<f4').reshape(total.shape).astype(float)
    gap=float((total.sum()-g.sum())*factor)
    result=dict(status='LINEAGE_DIAGNOSTIC_NOT_PRODUCTION',job=job,
        partition_max_error_fraction_peak=closure,baseline_max_difference_fraction_peak=delta,
        baseline_total_ratio=float(total.sum()/baseline.sum()),MeV_per_primary=energies,
        gpu_MeV_per_primary=float(g.sum()*factor),topas_minus_gpu_MeV_per_primary=gap,
        dose_pins={str(f):sha(f) for f in [gpu,*[root/f'{k}.bin' for k in GROUPS]]},limitations=m['limitations'])
    with (root/'analysis.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    np.savez(root/'depth_partition.npz',**{k:v.sum(axis=(1,2))*factor for k,v in data.items()})
    print(json.dumps(result,indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--job',type=int);a=p.parse_args();root=a.out.resolve()
    if a.job:analyze(root,a.job)
    else:prepare(root)

if __name__=='__main__':main()
