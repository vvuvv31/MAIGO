from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=Path(__file__).resolve().parents[2];root=r/'scratch/unified_joint_em_dense_20260913';rows=[]
for d in sorted((root/'transport').iterdir()):
 status=json.load(open(d/'status.json'));assert status['complete'] and status['quality']['queue_overflow_count']==0
 ref=r/'benchmark/benchmark20260913'/d.name;geom=json.load(open(ref/'geometry.json'));shape=tuple(geom['shape']);rho=np.zeros(shape)
 for b in geom['blocks']:rho[b['z0']:b['z1'],:,b['x0']:b['x1']]=b['rho']
 dose=np.fromfile(d/'dose_gpu.raw',dtype='<f4').reshape(shape);g=(dose*rho).sum((1,2))*.5e-6*6.241509074e12/status['histories'];z,t,old=np.loadtxt(ref/'idd.csv',delimiter=',').T
 row=dict(case=d.name,gpu_histories=status['histories'],topas_histories=geom['histories'],topas_peak_mm=z[t.argmax()],gpu_peak_mm=z[g.argmax()],peak_error_percent=100*(g.max()/t.max()-1),previous_peak_error_percent=100*(old.max()/t.max()-1),energy_error_percent=100*(g.sum()/t.sum()-1),balance_error=status['quality']['relative_energy_residual'])
 rows.append(row);np.savetxt(d/'idd.csv',np.column_stack([z,t,g]),delimiter=',',header='z_mm,topas_per_primary,gpu_per_primary')
 fig,ax=plt.subplots(2,1,figsize=(9,7),sharex=True)
 for y,label in [(t,'TOPAS default (200k)'),(old,'Previous GPU (200k)'),(g,'Unified all-ion EM (50k)')]:ax[0].plot(z,y,label=label)
 ax[0].legend();ax[0].set_ylabel('MeV / primary / 0.5 mm');ax[0].set_title(d.name)
 mask=t>0
 ax[1].plot(z[mask],100*(g[mask]/t[mask]-1));ax[1].set_ylim(-5,5);ax[1].set_ylabel('Relative error (%)');ax[1].set_xlabel('Depth (mm)');ax[1].set_xlim(0,1.2*z[t.argmax()])
 for a in ax:a.grid(alpha=.25)
 fig.tight_layout();fig.savefig(d/(d.name+'_idd.png'),dpi=160);plt.close(fig)
with (root/'transport_summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
print(json.dumps(rows,indent=2))
