from pathlib import Path
import numpy as np,json,math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=Path('/mnt/sdb/wuwei/MAIGO_review_fix_20260923');o=r/'evidence/percent1_20260923';d=o/'topas_phase_2m'
assert (d/'quality.json').exists()
t=np.loadtxt(d/'water_entrance.phsp');g=np.genfromtxt(r/'evidence/valley_diagnosis_20260923/entrance_cu005_s1.csv',delimiter=',',names=True)
assert np.all(t[:,6]==1) and np.all(t[:,13]==0) and np.all(abs(t[:,1]*10-60)<1e-4)
data={'TOPAS':{'x':t[:,0]*10,'E':t[:,5]/12,'dx':t[:,3],'dz':t[:,4],'N':2000000},'GPU':{'x':g['x_mm'],'E':g['kinetic_energy_MeV']/12,'dx':g['direction_x'],'dz':g['direction_z'],'N':1000000}}
sp=np.loadtxt(r/'data/stopping_power_water_geant4_11_3_2.csv',comments='#',delimiter=',',skiprows=2)
result={'scope':'Matched Cu 0.05 mm, same input physics/geometry; TOPAS 2M entrance-only run and GPU matched 1M s1. No cross-engine history joins. S(E)/abs(dz) is an unrestricted C12 diagnostic proxy, not scored total dose. Statistical errors here are independent-history approximations.','regions':{}}
fig,ax=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
for region in ['all','peak','valley']:
 rows={}
 for engine,a in data.items():
  mask=np.ones(len(a['x']),bool) if region=='all' else abs(a['x'])<.25 if region=='peak' else abs(abs(a['x'])-1.8)<.45
  E=a['E'][mask];dx=a['dx'][mask];dz=a['dz'][mask];n=len(E);N=a['N'];w=np.interp(E,sp[:,0],sp[:,1])/abs(dz)
  rows[engine]={'count':n,'fluence_per_source':n/N,'fluence_SE':math.sqrt((n/N)*(1-n/N)/N),'mean_MeVu':float(E.mean()),'mean_MeVu_SE':float(E.std(ddof=1)/math.sqrt(n)),'quantiles_MeVu':np.quantile(E,[.01,.1,.5,.9,.99]).tolist(),'theta_x_mrad_RMS':float(np.sqrt(np.mean(np.arctan2(dx,dz)**2))*1000),'S_over_dz_per_source':float(w.sum()/N),'S_over_dz_SE':float(np.sqrt((np.sum(w*w)-w.sum()**2/N)/(N*(N-1))))}
  if region=='valley':ax[0].hist(E,bins=np.arange(0,276,10),weights=np.full(n,1/N),histtype='step',label=engine,lw=1.5)
  if region=='all':ax[1].hist(a['x'],bins=np.arange(-8,8.01,.1),weights=np.full(len(a['x']),1/N),histtype='step',label=engine,lw=1.2)
 a,b=rows['GPU'],rows['TOPAS'];rows['GPU_vs_TOPAS']={'fluence_difference_percent':100*(a['fluence_per_source']/b['fluence_per_source']-1),'mean_energy_difference_MeVu':a['mean_MeVu']-b['mean_MeVu'],'mean_energy_difference_SE':math.hypot(a['mean_MeVu_SE'],b['mean_MeVu_SE']),'proxy_difference_percent':100*(a['S_over_dz_per_source']/b['S_over_dz_per_source']-1)}
 result['regions'][region]=rows;print(region,json.dumps(rows),flush=True)
ax[0].set(xlabel='C12 energy at water entrance (MeV/u)',ylabel='Particles / source / 10 MeV/u',title='Fixed central valleys at entrance')
ax[1].set(xlabel='Transverse x (mm)',ylabel='Particles / source / 0.1 mm',title='Water entrance C12 fluence',yscale='log',ylim=(1e-5,.06))
for a in ax:a.legend()
fig.savefig(o/'entrance_comparison.png',dpi=170);plt.close(fig)
(o/'entrance_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
