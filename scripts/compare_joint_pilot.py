#!/usr/bin/env python3
"""Dose screen for the hybrid CT pilot; no claim of clinical acceptance."""
import argparse,json,re
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('baseline',type=Path);p.add_argument('candidate',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True)
    def load(d):
        h=dict(l.split(' = ',1) for l in (d/'out/case/dose.mhd').read_text().splitlines() if ' = ' in l)
        v=np.memmap(d/'out/case'/h['ElementDataFile'],dtype='<f4',mode='r',shape=tuple(map(int,h['DimSize'].split()))[::-1])
        return v,h
    x,h=load(a.baseline);y,hy=load(a.candidate);assert h==hy
    execution=[json.loads((d/'execution.json').read_text()) for d in [a.baseline,a.candidate]]
    assert execution[0]['histories_per_batch']==execution[1]['histories_per_batch'] and execution[0]['seed']==execution[1]['seed']
    quality=[json.loads((d/'out/case/quality_report.json').read_text()) for d in [a.baseline,a.candidate]]
    assert all(q['accepted'] and q['queue_overflow_count']==0 for q in quality)
    logs=[(d/'gpu.log').read_text() for d in [a.baseline,a.candidate]]
    audit_prefixes=('[urban-v2-steps]','[unified-em-audit]','COPPER_PROTON_LOSS_QUALITY','AIR_IONIZATION_CLOCK','[all-ion-elastic]','[minibeam-water] delta')
    audits=[[l for l in log.splitlines() if l.startswith(audit_prefixes)] for log in logs]
    counters=dict((k,int(v)) for k,v in re.findall(r'(\w+)=(\d+)',next(l for l in logs[1].splitlines() if l.startswith('[electron-joint-pilot-work]'))))
    tasks=int(re.search(r'\[electron-queue-time\].*?tasks=(\d+)',logs[1])[1])
    result=dict(status='EXPERIMENTAL_CT_SCREEN_NOT_ACCEPTANCE',histories=execution[0]['histories_per_batch'],quality=quality,work=counters,
                started_fraction=counters['started']/tasks,total_packets=tasks,primary_em_audits_equal=audits[0]==audits[1],
                dose_sum_change_pct=100*(y.sum(dtype=float)/x.sum(dtype=float)-1),roi=[],pointwise=[])
    result['timing']=[dict(wall_seconds=e['wall_seconds'],electron_seconds=float(re.search(r'\[electron-queue-time\] seconds=([\d.eE+-]+)',l)[1]),
       transport_seconds=float(re.search(r'\[runtime-stage\] transport_total seconds=([\d.eE+-]+)',l)[1]),
       setup_seconds=float(re.search(r'\[runtime-stage\] transport_setup_including_upload seconds=([\d.eE+-]+)',l)[1])) for e,l in zip(execution,logs)]
    peak=float(x.max())
    for cutoff in [.001,.01,.1]:
        count=passed=0;worst=0.
        for iz in range(x.shape[0]):
            u=x[iz].astype(float);v=y[iz].astype(float);mask=u>cutoff*peak
            r=np.abs(v[mask]-u[mask])/u[mask];count+=len(r);passed+=int((r<=.03).sum());worst=max(worst,float(r.max(initial=0)))
        result['pointwise'].append(dict(reference_max_cutoff=cutoff,voxels=count,local_3pct_0mm_pass_pct=100*passed/count,max_local_pct=100*worst))
    origin=np.array(list(map(float,h['Offset'].split())));spacing=np.array(list(map(float,h['ElementSpacing'].split())))
    xx=origin[0]+np.arange(x.shape[2])*spacing[0];yy=origin[1]+np.arange(x.shape[1])*spacing[1];zz=origin[2]+np.arange(x.shape[0])*spacing[2]
    distance=np.abs(xx-np.rint(xx/3.6)*3.6);field=np.abs(xx)<=15
    masks={'peak':field&(distance<=.25),'valley':field&(distance>=1.55)}
    iy=np.flatnonzero(np.abs(yy)<=15);profiles=[]
    for start,end in [(0,20),(20,80),(80,120),(120,180),(180,230),(230,303.5)]:
        iz=np.flatnonzero((zz>=start)&(zz<end))
        pu=x[np.ix_(iz,iy,np.arange(x.shape[2]))].sum(axis=(0,1),dtype=float)
        pv=y[np.ix_(iz,iy,np.arange(y.shape[2]))].sum(axis=(0,1),dtype=float)
        profiles.append((start,end,pu,pv))
        profile_mask=field&(pu>.01*pu[field].max())
        result.setdefault('profile_local',[]).append(dict(depth_mm=[start,end],
            max_local_pct=float((abs(pv[profile_mask]-pu[profile_mask])/pu[profile_mask]).max()*100)))
        for name,mask in masks.items():
            u=float(pu[mask].sum());v=float(pv[mask].sum());result['roi'].append(dict(depth_mm=[start,end],region=name,reference=u,candidate=v,change_pct=100*(v/u-1) if u else None))
    np.savez(a.output/'profiles.npz',x_mm=xx,baseline=np.array([t[2] for t in profiles]),candidate=np.array([t[3] for t in profiles]))
    (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(13,10),layout='constrained')
    for ax,(start,end,pu,pv) in zip(axes.flat,profiles):
        ax.plot(xx[field],pu[field],label='Legacy');ax.plot(xx[field],pv[field],label='Joint pilot',alpha=.8)
        ax.set_title(f'Depth {start}–{end} mm');ax.set_xlabel('x (mm)');ax.set_ylabel('Dose sum (Gy)');ax.legend()
    fig.savefig(a.output/'profiles.png',dpi=150)
    print(json.dumps({k:result[k] for k in ['started_fraction','work','dose_sum_change_pct','roi','pointwise','timing']},indent=2))
if __name__=='__main__':main()
