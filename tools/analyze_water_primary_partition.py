"""Compare same-run partitions; primary definitions are bounded, not equated."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
from prepare_water_primary_partition import GROUPS
from analyze_water_lineage_roundoff import closes
from run_topas10x_gpu_benchmark import sha

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--job',type=int,required=True)
    a=p.parse_args();root=a.out.resolve();m=json.loads((root/'manifest.json').read_text())
    status=subprocess.check_output(['sacct','-j',str(a.job),'--format=JobIDRaw,State,ExitCode','-n','-P'],text=True)
    if f'{a.job}|COMPLETED|0:0' not in status.splitlines():raise ValueError('Incomplete TOPAS')
    for name,h in m['pins'].items():
        if sha(Path(name))!=h:raise ValueError('Changed frozen input '+name)
    files=[];t={}
    for name in GROUPS:
        f=root/f'{name}.bin';files.append(f)
        h=f.with_suffix('.binheader').read_text()
        for line in ('# X in 64 bins of 0.2 cm','# Y in 64 bins of 0.2 cm','# Z in 800 bins of 0.05 cm','DoseToMedium ( Gy ) : Sum'):
            if line not in h:raise ValueError('TOPAS grid/units mismatch')
        t[name]=np.fromfile(f,'<f8').reshape(800,64,64)
    if not closes(t['total'],[v for k,v in t.items() if k!='total']):raise ValueError('TOPAS partition fails closure')
    baseline=np.fromfile(m['baseline'],'<f8').reshape(800,64,64)
    if not np.array_equal(t['total'],baseline):raise ValueError('TOPAS total changed')
    gpu=Path('/mnt/sda/wuwei/unified_water300_origin_20260907');g={}
    for h in sorted(gpu.glob('origin_*.mhd')):
        text=h.read_text()
        for line in ('DimSize = 64 64 800','ElementSpacing = 2 2 0.5','Offset = -63 -63 0.25'):
            if line not in text:raise ValueError('GPU origin geometry mismatch')
        f=h.with_suffix('.raw');files.extend([h,f]);g[h.stem.removeprefix('origin_')]=np.fromfile(f,'<f4').reshape(800,64,64).astype(float)
    if len(g)!=8:raise ValueError('Incomplete GPU categories')
    gtotal=np.fromfile(gpu/'dose.raw','<f4').reshape(800,64,64).astype(float)
    if not closes(gtotal,list(g.values())):raise ValueError('GPU partition fails closure')
    old=Path('/mnt/sda/wuwei/unified_water300_seed2_20260907/dose.raw')
    if not np.array_equal(gtotal,np.fromfile(old,'<f4').reshape(gtotal.shape)):raise ValueError('GPU total changed')
    factor=2e-6*6.241509074e12/50000
    te={k:float(v.sum()*factor) for k,v in t.items()};ge={k:float(v.sum()*factor) for k,v in g.items()}
    result=dict(status='PRIMARY_PARTITION_DIAGNOSTIC_NOT_PRODUCTION',job=a.job,
        topas_MeV_per_primary=te,gpu_MeV_per_primary=ge,
        gpu_primary_minus_topas_primary=ge['primary']-te['primary'],
        gpu_primary_minus_topas_primary_plus_all_electrons=ge['primary']-te['primary']-te['electrons_no_neutral'],
        topas_other_secondary_minus_gpu_secondary=te['other_secondary_no_neutral']-sum(v for k,v in ge.items() if k!='primary'),
        total_unchanged_both=True,closure_pass_both=True,
        pins={str(f):sha(f) for f in [*files,old,gpu/'dose.raw',Path(__file__).resolve()]},limitations=m['limitations'])
    with (root/'analysis.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    np.savez(root/'depth_partitions.npz',**{'topas_'+k:v.sum(axis=(1,2))*factor for k,v in t.items()},
             **{'gpu_'+k:v.sum(axis=(1,2))*factor for k,v in g.items()})
    print(json.dumps({k:v for k,v in result.items() if k!='pins'},indent=2))

if __name__=='__main__':main()
