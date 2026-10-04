#!/usr/bin/env python3
"""First C12 CT minibeam comparison; BODY y-mean is explicitly 2D gamma."""
from pathlib import Path
import json,struct
import numpy as np
import pymedphys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'out/carbon_ct_minibeam_20261002'

def main():
    base=json.loads((ROOT/'out/ct_topas_gamma_20261001/reference_manifest.json').read_text())
    nx,ny,nz=base['shape_xyz'];shape=(nz,ny,nx)
    x=-60+(np.arange(nx)+.5)*.1;y=-42+(np.arange(ny)+.5)*2;z=(np.arange(nz)+.5)*.5
    ym=abs(y)<15
    body=np.memmap(base['body']['path'],np.uint8,'r',shape=shape)[:,ym,:].astype(bool)
    rho=np.memmap(ROOT/'inputs/lung_ct/ct.bin',np.float32,'r',offset=44,shape=shape)[:,ym,:]
    count=body.sum(1)
    raw={};total={};maps={}
    for engine in ['gpu','topas']:
        batches=[];histories=[]
        for i in range(2):
            if engine=='gpu':
                d=OUT/'gpu_batches'/f'batch{i:06d}'
                q=json.loads((d/'quality_report.json').read_text());assert q['accepted'] and q['queue_overflow_count']==0
                ledger=json.loads((d/'energy_ledger.json').read_text());n=int(ledger['histories'])
                a=np.asarray(np.memmap(d/'dose.raw',np.float32,'r',shape=shape)[:,ym,:],float)/n
            else:
                d=OUT/f'topas_b{i+1:02d}';s=json.loads((d/'status.json').read_text());assert s['accepted'];n=s['histories']
                a=np.asarray(np.memmap(d/'edep.bin',np.float64,'r',shape=shape)[:,ym,:],float)/n
                a*=1.602176634e-13/(rho*1e-7)
            assert np.isfinite(a).all() and (a>=0).all()
            batches.append(a*body);histories.append(n)
        assert len(set(histories))==1
        raw[engine]=batches;total[engine]=sum(histories)
        maps[engine]=np.divide(np.mean(batches,axis=0).sum(1),count,out=np.zeros((nz,nx)),where=count>0)
    r,g=maps['topas'],maps['gpu'];mask=(count>0)&(abs(x)[None,:]<15)&(r>=.03*r.max())
    summary=dict(status='INITIAL_COMPARISON_NOT_FINAL_ACCEPTANCE',histories=total,
        method='2D BODY y-mean, |y|<15mm; evaluated |x|<15mm; TOPAS map max 3% threshold; no fitted scaling',
        limitation='Two 100k batches per engine are an initial noise-limited comparison, not a converged dose claim.',gamma=[],rois=[])
    for dd,dta in [(1,.2),(2,.2),(3,.2),(3,0)]:
        for local in [False,True]:
            if dta:
                ga=pymedphys.gamma((z,x),r,(z,x),g,dd,dta,local_gamma=local,
                    lower_percent_dose_cutoff=3,interp_fraction=10,max_gamma=1.01,
                    skip_once_passed=True,ram_available=1024**3,interp_algo='scipy')
            else:
                den=dd*.01*(r if local else r.max())
                ga=np.divide(abs(g-r),den,out=np.full_like(r,np.inf),where=den>0)
            assert np.isfinite(ga[mask]).all()
            row=dict(dose_percent=dd,distance_mm=dta,local=local,pass_percent=float(100*np.mean(ga[mask]<=1)),evaluated=int(mask.sum()))
            summary['gamma'].append(row);print(row,flush=True)
    fold=(x+1.8)%3.6-1.8
    for name,xmask in [('central_peak',abs(x)<.2),('pooled_valleys',(abs(fold)>1.6)&(abs(x)<12.6)),('field',abs(x)<15)]:
        region=body[:,:,xmask]
        values={e:np.array([a[:,:,xmask].sum()/region.sum() for a in batches]) for e,batches in raw.items()}
        ratio=values['gpu'].mean()/values['topas'].mean()
        se={e:v.std(ddof=1)/np.sqrt(len(v)) for e,v in values.items()}
        summary['rois'].append(dict(name=name,scope='all BODY depth',bias_percent=float(100*(ratio-1)),
            batch_sem_pp=float(100*np.hypot(se['gpu'],ratio*se['topas'])/values['topas'].mean())))
    (OUT/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
    np.savez(OUT/'comparison_maps.npz',topas=r,gpu=g,x=x,z=z,body_counts=count)
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for ax,(name,xmask) in zip(axes,[('Central peak',abs(x)<.2),('Pooled valleys',(abs(fold)>1.6)&(abs(x)<12.6))]):
        den=body[:,:,xmask].sum((1,2))
        for engine in ['topas','gpu']:
            line=np.mean(raw[engine],axis=0)[:,:,xmask].sum((1,2))
            ax.plot(z,np.divide(line,den,out=np.zeros(nz),where=den>0),label=engine.upper())
        ax.set(title=name,xlabel='Depth (mm)',ylabel='BODY Gy / incident C12');ax.legend()
    fig.suptitle('Carbon 250 MeV/u CT minibeam: initial statistics, no fitted normalization')
    fig.savefig(OUT/'comparison.png',dpi=150)

if __name__=='__main__':main()
