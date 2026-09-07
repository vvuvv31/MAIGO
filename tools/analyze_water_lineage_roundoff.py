"""Lineage closure with bounds from TOPAS G4float fSum output, not dose fitting."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
from water_neutral_lineage import GROUPS
from run_topas10x_gpu_benchmark import sha

def closes(total,parts):
    # TsVBinnedScorer.hh: fSum is G4float even when PrintBinary writes G4double.
    # Four independently rounded nonnegative outputs: conservative sum of
    # relative rounding bounds. This is voxel-local, not a peak-dose allowance.
    bound=np.finfo(np.float32).eps*(np.abs(total)+sum(np.abs(p) for p in parts))
    bound+=4*np.finfo(np.float32).smallest_subnormal
    return bool(np.all(np.abs(sum(parts)-total)<=bound))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--job',type=int,required=True)
    a=p.parse_args();root=a.out.resolve();m=json.loads((root/'manifest.json').read_text())
    status=subprocess.check_output(['sacct','-j',str(a.job),'--format=JobIDRaw,State,ExitCode','-n','-P'],text=True)
    if f'{a.job}|COMPLETED|0:0' not in status.splitlines():raise ValueError('Incomplete job')
    for name,h in m['pins'].items():
        if sha(Path(name))!=h:raise ValueError('Changed input '+name)
    header=Path('/home/wuwei/topas/OpenTOPAS-4.2.3/scoring/TsVBinnedScorer.hh')
    if 'G4float fSum;' not in header.read_text():raise ValueError('Output-rounding assumption changed')
    d={}
    for k in GROUPS:
        h=(root/f'{k}.binheader').read_text()
        for required in ('# X in 64 bins of 0.2 cm','# Y in 64 bins of 0.2 cm','# Z in 800 bins of 0.05 cm','DoseToMedium ( Gy ) : Sum'):
            if required not in h:raise ValueError('Header mismatch')
        d[k]=np.fromfile(root/f'{k}.bin','<f8').reshape(800,64,64)
        if not np.isfinite(d[k]).all() or d[k].min()<0:raise ValueError('Invalid payload')
    total=d['total'];parts=[d[k] for k in GROUPS if k!='total']
    if total.max()<=0 or not closes(total,parts):raise ValueError('Partition exceeds output-rounding bound')
    baseline=np.fromfile(m['baseline'],'<f8').reshape(total.shape)
    if not np.array_equal(total,baseline):raise ValueError('Added scorers changed total dose')
    gpu=Path('/mnt/sda/wuwei/unified_water300_seed2_20260907/dose.raw')
    gm=json.loads((gpu.parent/'manifest.json').read_text())
    if sha(gpu)!=gm['dose_sha256']:raise ValueError('GPU dose changed')
    f=2e-6*6.241509074e12/50000;g=np.fromfile(gpu,'<f4').astype(float).sum()*f
    energies={k:float(v.sum()*f) for k,v in d.items()}
    gap=energies['total']-g;neutral=energies['neutron']+energies['gamma_no_neutron']
    result=dict(status='LINEAGE_DIAGNOSTIC_NOT_PRODUCTION',job=a.job,baseline_bitwise_equal=True,
        MeV_per_primary=energies,gpu_MeV_per_primary=float(g),gap_MeV_per_primary=float(gap),
        neutron_gamma_lineage_fraction_of_gap=neutral/gap,
        neither_minus_gpu_MeV_per_primary=energies['neither']-g,
        max_partition_error_fraction_peak=float(np.max(np.abs(sum(parts)-total))/total.max()),
        closure_rule='Per-voxel IEEE float32 output rounding bound; G4float fSum stored as double',
        pins={str(x):sha(x) for x in [header,Path(__file__).resolve(),gpu,*[root/f'{k}.bin' for k in GROUPS]]},
        limitations=m['limitations'])
    with (root/'analysis.json').open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    np.savez(root/'depth_partition.npz',**{k:v.sum(axis=(1,2))*f for k,v in d.items()})
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
