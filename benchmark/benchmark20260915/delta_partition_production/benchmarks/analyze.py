"""Check paired 3D outputs, assemble block scorers, compare mass-weighted IDD/profiles."""
from pathlib import Path
import json,re,csv,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sigma_fit import curves
ROOT=Path(__file__).resolve().parent

def analyze(d):
 geom=json.loads((d/'geometry.json').read_text());n=geom['histories'];shape=tuple(geom['shape'])
 status=json.loads((d/'gpu_status.json').read_text());assert status['complete'] and status['quality']['queue_overflow_count']==0
 ledger=json.loads((d/'out/gpu/energy_ledger.json').read_text());assert ledger['histories']==n
 log=(d/'topas.log').read_text();assert 'Finalization:' in log
 assert sum(map(int,re.findall(r'Particle source .*?: Total number of histories: (\d+)',log)))==n
 assert 'Max kinetic energy for tables                      10 GeV' in log
 t=np.zeros(shape);rho=np.zeros(shape)
 for j,b in enumerate(geom['blocks']):
  p=d/f'dose{j}.bin';h=p.with_suffix('.binheader').read_text()
  dims=[int(re.search(rf'# {a} in (\d+) bins',h)[1]) for a in 'XYZ'];expected=[b['x1']-b['x0'],160,b['z1']-b['z0']]
  assert dims==expected and 'DoseToMedium ( Gy ) : Sum' in h
  spacing=[float(re.search(rf'# {a} in \d+ bins of ([0-9.]+) cm',h)[1])*10 for a in 'XYZ'];assert spacing==[1.,1.,.5]
  arr=np.fromfile(p,dtype='<f8').reshape(dims[::-1]);sl=np.s_[b['z0']:b['z1'],:,b['x0']:b['x1']];t[sl]=arr;rho[sl]=b['rho']
 g=np.fromfile(d/'dose_gpu.raw',dtype='<f4').reshape(shape)
 mhd=(d/'dose_gpu.mhd').read_text();assert 'Offset = -79.5 -79.5 0.25' in mhd and 'ElementSpacing = 1 1 0.5' in mhd
 assert np.all(rho>0)
 for a in (t,g):assert np.isfinite(a).all() and a.min()>=0 and a.max()>0
 # Gy * voxel mass (kg) * MeV/J / primary count, then lateral sum.
 factor=.5e-6*6.241509074e12/n
 te=t*rho*factor;ge=g*rho*factor;td=te.sum((1,2));gd=ge.sum((1,2));z=np.arange(700)*.5+.25;x=np.arange(160)-79.5
 np.savetxt(d/'idd.csv',np.column_stack((z,td,gd)),delimiter=',',header='z_mm,topas_MeV_per_primary_per_0.5mm,gpu_MeV_per_primary_per_0.5mm')
 fig,ax=plt.subplots(2,1,figsize=(10,7),sharex=True)
 for label,v in [('TOPAS',td),('GPU',gd)]:ax[0].plot(z,v,label=label)
 ax[0].legend();ax[0].set_ylabel('MeV / primary / 0.5 mm');ax[0].set_title(d.name)
 ax[1].plot(z,100*(gd-td)/td.max());ax[1].set_ylabel('Difference / TOPAS peak (%)');ax[1].set_xlabel('Depth (mm)')
 for a in ax:a.grid(alpha=.25)
 fig.tight_layout();plt.close(fig)
 depths=[10,40,80,120,160,200,240,280,320] if d.name.startswith('b4') else [float(z[td.argmax()])*f for f in (.1,.3,.5,.7,.9,1.,1.05,1.1,1.2)]
 fig,axes=plt.subplots(3,3,figsize=(14,11));profiles=[]
 for a,dep in zip(axes.ravel(),depths):
  mask=np.abs(z-dep)<1
  for label,arr in [('TOPAS',te),('GPU',ge)]:
   y=arr[mask].sum((0,1));a.plot(x,y,label=label)
   profiles.extend((dep,float(xx),label,float(yy)) for xx,yy in zip(x,y))
  a.set_title(f'z = {dep:.2f} mm, 2 mm window');a.set_xlabel('x (mm)');a.set_ylabel('MeV / primary / x bin');a.grid(alpha=.2)
 axes[0,0].legend();fig.tight_layout();plt.close(fig)
 with (d/'profiles.csv').open('w') as f:w=csv.writer(f);w.writerow(['depth_mm','x_mm','code','MeV_per_primary']);w.writerows(profiles)
 if not d.name.startswith('b4'):
  # Fixed 2mm windows; fits are descriptive. Invalid components remain explicitly flagged.
  xmax=min(350,float(z[td.argmax()])*1.2);tc=curves(te,xmax);gc=curves(ge,xmax)
  (d/'sigma_fits.json').write_text(json.dumps(dict(topas=tc,gpu=gc),indent=2))
  fig,axes=plt.subplots(1,2,figsize=(12,4))
  for a,key in zip(axes,['core','halo']):
   for label,c in [('TOPAS',tc),('GPU',gc)]:
    zz=[v['depth_mm'] for v in c];v=[v['fit'][key] if v['fit'] and v['fit']['valid'] else np.nan for v in c];a.plot(zz,v,label=label)
   a.set_xlabel('Depth (mm)');a.set_ylabel(f'Sigma {key} (mm)');a.legend();a.grid(alpha=.2)
  fig.tight_layout();plt.close(fig)
 result=dict(case=d.name,topas_peak_mm=float(z[td.argmax()]),gpu_peak_mm=float(z[gd.argmax()]),topas_MeV_per_primary=float(td.sum()),gpu_MeV_per_primary=float(gd.sum()),energy_ratio_GPU_TOPAS=float(gd.sum()/td.sum()),IDD_RMSE_percent_TOPAS_peak=float(100*np.sqrt(np.mean((gd-td)**2))/td.max()),gpu_elapsed_s=status['elapsed_s'])
 ts=json.loads((d/'topas_status.json').read_text());assert ts['complete']
 result['topas_elapsed_s']=ts['elapsed_s'];result['wall_time_ratio_TOPAS_GPU']=ts['elapsed_s']/status['elapsed_s']
 result['gpu_voxel_energy_to_ledger_ratio']=float(ge.sum()*n/ledger['E_dep_in_grid_MeV'])
 assert abs(result['gpu_voxel_energy_to_ledger_ratio']-1)<1e-4
 (d/'comparison.json').write_text(json.dumps(result,indent=2));return result
if __name__=='__main__':
 rows=[analyze(ROOT/c[0]) for c in json.loads((ROOT/'cases.json').read_text())]
 with (ROOT/'summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
