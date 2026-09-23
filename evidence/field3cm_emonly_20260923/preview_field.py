from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=Path('/mnt/sdb/wuwei/MAIGO_review_fix_20260923')
O=R/'evidence/field3cm_emonly_20260923';D=O/'results/preview_b1';D.mkdir(exist_ok=True)
M=json.loads((O/'manifest.json').read_text());P=json.loads((O/'preflight.json').read_text())
td=Path(M['cases']['topas_b1']);gd=Path(M['cases']['gpu_b1']).parent
tq=json.loads((td/'quality.json').read_text());gq=json.loads((gd/'quality.json').read_text())
n=M['histories_per_batch']
assert P['accepted'] and tq['accepted'] and gq['accepted'] and tq['histories']==gq['histories']==n
assert tq['topas_binary_sha256']==P['topas_binary_sha256'] and gq['gpu_binary_sha256']==M['gpu_binary_sha256']
t=np.fromfile(td/'dose.bin',dtype='<f8').reshape(1000,1000)/n
g=np.fromfile(gq['dose_path'],dtype='<f4').reshape(1000,1000).astype(float)/n
x=-49.95+np.arange(1000)*.1;z=.125+np.arange(1000)*.25
rows=[];curves={}
fig,axs=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
for col,(label,mask) in enumerate([('Peak',abs(x)<.25),('Valley',abs(abs(x)-1.8)<.45)]):
    a=g[:,mask].mean(1);b=t[:,mask].mean(1)
    r=np.divide(a,b,out=np.full_like(a,np.nan),where=b>0)*100-100
    curves[label]=(a,b,r)
    ax=axs[0,col];ax.plot(z,b*1e9,label='TOPAS');ax.plot(z,a*1e9,label='GPU',ls='--')
    ax.set(title=label,xlim=(0,140),ylim=(0,None),xlabel='Water depth (mm)',ylabel='nGy / source');ax.legend()
    ax=axs[1,col];valid=(z<=125)&(b>b.max()*.01)
    ax.axhspan(-1,1,color='#d7efda');ax.axhline(0,color='gray',lw=.5);ax.plot(z[valid],r[valid],lw=.8)
    ax.set(xlim=(0,125),xlabel='Water depth (mm)',ylabel='(GPU/TOPAS - 1) (%)',title='Preliminary difference; no statistical CI')
for d in [20,40,60,80,100,120]:
    k=int(np.argmin(abs(z-d)));rows.append(dict(depth_mm=float(z[k]),**{s.lower()+'_difference_percent':float(c[2][k]) for s,c in curves.items()}))
fig.suptitle('PRELIMINARY: 3 cm x 3 cm field | EM-only | 3.2M histories per engine\nFirst independent batch only; 100 mm vertical averaging; statistical uncertainty not yet estimated')
fig.savefig(D/'peak_valley_depth_preview.png',dpi=160);plt.close(fig)
summary=dict(status='PRELIMINARY_ONE_BATCH',histories_per_engine=n,rows=rows,
             total_deposited_energy_difference_percent=float((g.sum()/t.sum()-1)*100),
             note='One batch per engine: no independent-batch confidence interval. Final 4-batch analysis follows automatically.')
(D/'preview.json').write_text(json.dumps(summary,indent=2)+'\n')
print('PREVIEW',json.dumps(summary),flush=True)
