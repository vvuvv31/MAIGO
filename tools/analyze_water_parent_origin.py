"""Audit existing parent-origin scorer against GPU, retaining semantic caveats."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
from analyze_water_lineage_roundoff import closes
from run_topas10x_gpu_benchmark import sha

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--job',type=int,required=True)
    a=p.parse_args();root=a.out.resolve();m=json.loads((root/'manifest.json').read_text())
    status=subprocess.check_output(['sacct','-j',str(a.job),'--format=JobIDRaw,State,ExitCode','-n','-P'],text=True)
    if f'{a.job}|COMPLETED|0:0' not in status.splitlines():raise ValueError('Incomplete TOPAS')
    for f,h in m['pins'].items():
        if sha(Path(f))!=h:raise ValueError('Frozen input changed: '+f)
    data={};files=[]
    for name in ['total','neutral_lineage',*m['mapping']]:
        f=root/f'{name}.bin';h=f.with_suffix('.binheader').read_text()
        for expected in ['# X in 64 bins of 0.2 cm','# Y in 64 bins of 0.2 cm','# Z in 800 bins of 0.05 cm','( Gy ) : Sum']:
            if expected not in h:raise ValueError('Wrong grid/quantity')
        data[name]=np.fromfile(f,'<f8').reshape(800,64,64);files.append(f)
        if not np.isfinite(data[name]).all() or data[name].min()<0:raise ValueError('Invalid data')
    if not closes(data['total'],[v for k,v in data.items() if k!='total']):raise ValueError('Parent partition fails closure')
    baseline=np.fromfile(m['baseline'],'<f8').reshape(800,64,64)
    change=float(np.max(np.abs(data['total']-baseline))/baseline.max())
    if change>2*np.finfo(np.float32).eps:raise ValueError('Total changed beyond output rounding')
    factor=2e-6*6.241509074e12/50000
    energies={k:float(v.sum()*factor) for k,v in data.items()}
    gpu=Path('/mnt/sda/wuwei/unified_water300_origin_20260907')
    comparison={}
    for topas,label in m['mapping'].items():
        if label in ('unclassified','other_neutral'):continue
        f=gpu/f'origin_{label}.raw';g=np.fromfile(f,'<f4').reshape(800,64,64).astype(float);files.append(f)
        gsum=float(g.sum()*factor);tsum=energies[topas]
        comparison[topas]=dict(topas=tsum,gpu=gsum,gpu_minus_topas=gsum-tsum,
                              gpu_over_topas=gsum/tsum if tsum>0 else None)
    result=dict(status='PARENT_ORIGIN_DIAGNOSTIC_NOT_PRODUCTION',job=a.job,
        total_bitwise_equal=bool(np.array_equal(data['total'],baseline)),total_change_fraction_peak=change,
        partition_closure=True,topas_MeV_per_primary=energies,comparison_MeV_per_primary=comparison,
        unclassified_fraction_total=energies['unclassified']/energies['total'],
        pins={str(f):sha(f) for f in [*files,Path(__file__).resolve()]},limitations=m['limitations'])
    with (root/'analysis.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    np.savez(root/'depth_partitions.npz',**{k:v.sum(axis=(1,2))*factor for k,v in data.items()})
    print(json.dumps({k:v for k,v in result.items() if k!='pins'},indent=2))

if __name__=='__main__':main()
