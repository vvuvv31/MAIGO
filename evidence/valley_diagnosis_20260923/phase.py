from pathlib import Path
import numpy as np,json
r=Path('/mnt/sdb/wuwei/MAIGO_review_fix_20260923');o=r/'evidence/valley_diagnosis_20260923'
e=np.genfromtxt(o/'entrance_s1.csv',delimiter=',',skip_header=1,usecols=(0,4,5,7,9,11,12))
p=np.genfromtxt(o/'planes_s1.csv',delimiter=',',skip_header=1,usecols=(0,2,3,4,6,8,13))
assert len(np.unique(e[:,0]))==len(e)
assert np.all((e[:,5]>0)==(e[:,6]>0))
idx=np.full(int(max(e[:,0].max(),p[:,0].max()))+1,-1,dtype=int);idx[e[:,0].astype(int)]=np.arange(len(e))
assert np.all(idx[p[:,0].astype(int)]>=0)
sp=np.loadtxt(r/'data/stopping_power_water_geant4_11_3_2.csv',comments='#',delimiter=',',skiprows=2)
def summarize(energy,x,dx,dz,cu,path,weight):
 if len(energy)==0:return {'count':0}
 stopping=np.interp(energy/12,sp[:,0],sp[:,1]);proxy=weight*stopping/np.maximum(abs(dz),1e-12)
 def avg(v):return float(np.average(v,weights=weight))
 return {'count':len(energy),'weight':float(weight.sum()),'ever_copper_fluence_fraction':avg(cu),
         'ever_copper_local_dose_proxy_fraction':float(proxy[cu].sum()/proxy.sum()),
         'energy_MeVu_mean':avg(energy/12),'energy_MeVu_quantiles':np.quantile(energy/12,[.1,.5,.9]).tolist(),
         'x_mm_mean':avg(x),'theta_x_mrad_rms':float(np.sqrt(avg(np.arctan2(dx,dz)**2))*1000),
         'copper_path_mean_mm':avg(path),'copper_energy_MeVu_mean':float(np.average(energy[cu]/12,weights=weight[cu])) if np.any(cu) else None,
         'never_copper_energy_MeVu_mean':float(np.average(energy[~cu]/12,weights=weight[~cu])) if np.any(~cu) else None,
         'Ebelow100MeVu_fluence_fraction':avg(energy<1200)}
out={'scope':'GPU Cu 0.25 mm, seed 2026092801, 1M source primaries; actual same-run source_history join only',
     'dose_proxy_definition':'Unrestricted stopping S(E)/abs(direction_z) weighted plane crossings. Diagnostic approximation, not a separate dose tally and not a TOPAS comparison.','depths':{}}
for depth in [0,20,100,120]:
 if depth==0:
  ids=e[:,0].astype(int);energy=e[:,1];x=e[:,2];dx=e[:,3];dz=e[:,4];cu=e[:,5]>0;path=e[:,6];weight=np.ones(len(e))
 else:
  a=p[p[:,1]==depth];ids=a[:,0].astype(int);energy=a[:,2];x=a[:,3];dx=a[:,4];dz=a[:,5];weight=a[:,6]
  ent=e[idx[ids]];cu=ent[:,5]>0;path=ent[:,6]
 rows={}
 for name,mask in [('all',np.ones(len(x),bool)),('central_peak',abs(x)<.25),('central_valleys',abs(abs(x)-1.8)<.45)]:
  rows[name]=summarize(energy[mask],x[mask],dx[mask],dz[mask],cu[mask],path[mask],weight[mask])
 out['depths'][str(depth)]=rows
(o/'phase_attribution.json').write_text(json.dumps(out,indent=2)+'\n')
for depth,rows in out['depths'].items():
 print(depth, 'peak',rows['central_peak'],'valley',rows['central_valleys'],flush=True)
