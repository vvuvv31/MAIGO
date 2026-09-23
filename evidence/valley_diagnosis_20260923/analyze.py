from pathlib import Path
import numpy as np,json,csv,hashlib,importlib.util
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=Path('/mnt/sdb/wuwei/MAIGO_review_fix_20260923');o=r/'evidence/valley_diagnosis_20260923'
ref=Path('/mnt/sda/wuwei/minibeam_single_center_em_only_water75ev_e250_10m/topas_7378')
t=np.fromfile(ref/'dose.bin',dtype='<f8').reshape(1000,1000)/1e7
x=-49.95+np.arange(1000)*.1;z=.125+np.arange(1000)*.25
pm=abs(x)<.25;vm=abs(abs(x)-1.8)<.45
def stats(a):
 a=np.asarray(a,float);return {'mean':float(a.mean()),'se':float(a.std(ddof=1)/np.sqrt(len(a))) if len(a)>1 else None,'values':a.tolist()}
def ratio(g,ref):
 s=stats(g);return {'gpu':s,'topas':float(ref),'relative_difference':s['mean']/ref-1,'gpu_only_ratio_se':s['se']/ref}
def load(stem,n):return np.stack([np.fromfile(r/f'out/{stem}_s{i}/dose.raw',dtype='<f4').reshape(1000,1000).astype(float)/n for i in [1,2,3]])
data={'previous_250k':load('boundary_final_fullchain',250000),'highstat_1m':load('valley_stats1m',1000000),'matched_cu005_250k':load('valley_cu005',250000)}
if all((r/f'out/valley_cu0051m_s{i}/dose.raw').exists() for i in [1,2,3]):data['matched_cu005_1m']=load('valley_cu0051m',1000000)
spec=importlib.util.spec_from_file_location('gate',r/'evidence/urban_d51e599_20260923/dose_gate.py');gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
j={'status':'EXPLORATORY_TOPAS_REFERENCE_SE_MISSING','definition':'Fixed central valleys abs(abs(x)-1.8)<0.45 mm, peak abs(x)<0.25 mm; no normalization fitting; SE from 3 independent GPU runs only','cases':{}}
for name,g in data.items():
 case={'energy_MeV_per_source':ratio(g.sum((1,2))*2.5e-6/1.602176634e-13,float(t.sum()*2.5e-6/1.602176634e-13)),'R80_mm':{'gpu':stats([gate._r80(a.sum(1),z) for a in g]),'topas':gate._r80(t.sum(1),z)},'rows':[]}
 for dep in [20,40,60,80,100,120]:
  iz=int(np.argmin(abs(z-dep)))
  for width in [.25,2.5,10]:
   mask=np.arange(1000)==iz if width==.25 else abs(z-dep)<width/2
   vg=g[:,mask][:,:,vm].mean((1,2));vt=t[mask][:,vm].mean();pg=g[:,mask][:,:,pm].mean((1,2));pt=t[mask][:,pm].mean()
   case['rows'].append({'depth_mm':dep,'window_width_mm':width,'valley':ratio(vg,vt),'peak':ratio(pg,pt),
       'valley_excess_relative_to_peak':float((vg.mean()-vt)/pt),'plane_sum':ratio(g[:,mask].sum((1,2)),t[mask].sum()),
       'right_valley':ratio(g[:,mask][:,:,vm&(x>0)].mean((1,2)),t[mask][:,vm&(x>0)].mean()),
       'left_valley':ratio(g[:,mask][:,:,vm&(x<0)].mean((1,2)),t[mask][:,vm&(x<0)].mean())})
   case['rows'][-1]['central_fwhm_mm']=ratio([gate._fwhm_1d(a[mask].mean(0),x) for a in g],gate._fwhm_1d(t[mask].mean(0),x))
 j['cases'][name]=case
 print(name,'R80',case['R80_mm'],'ENERGY',case['energy_MeV_per_source']['relative_difference'],flush=True)
 for a in case['rows']:
  print(a['depth_mm'],a['window_width_mm'],'valley%',round(100*a['valley']['relative_difference'],3),'SEpp',round(100*a['valley']['gpu_only_ratio_se'],3),'peak%',round(100*a['peak']['relative_difference'],3),flush=True)
if 'matched_cu005_1m' in j['cases']:
 pairs=[]
 for a,b in zip(j['cases']['highstat_1m']['rows'],j['cases']['matched_cu005_1m']['rows']):
  changes=(np.asarray(b['valley']['gpu']['values'])-a['valley']['gpu']['values'])/a['valley']['topas']
  pairs.append({'depth_mm':a['depth_mm'],'window_width_mm':a['window_width_mm'],'paired_valley_change_in_reference_units':stats(changes)})
 j['paired_copper_step_change']=pairs
with (o/'valley_comparison.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['case','depth_mm','depth_window_mm','valley_relative_percent','GPU_only_SE_pp','peak_relative_percent','excess_as_percent_TOPAS_peak','plane_sum_relative_percent'])
 for name,case in j['cases'].items():
  for a in case['rows']:w.writerow([name,a['depth_mm'],a['window_width_mm'],100*a['valley']['relative_difference'],100*a['valley']['gpu_only_ratio_se'],100*a['peak']['relative_difference'],100*a['valley_excess_relative_to_peak'],100*a['plane_sum']['relative_difference']])
(o/'comparison.json').write_text(json.dumps(j,indent=2)+'\n')
fig,axes=plt.subplots(2,2,figsize=(12,9),constrained_layout=True)
colors={'previous_250k':'#94a3b8','highstat_1m':'#dc2626','matched_cu005_250k':'#2563eb'}
labels={'previous_250k':'Cu 0.25 mm: previous 250k x 3','highstat_1m':'Cu 0.25 mm: 1M x 3','matched_cu005_250k':'Cu 0.05 mm: paired 250k x 3'}
colors['matched_cu005_1m']='#15803d';labels['matched_cu005_1m']='Cu 0.05 mm: paired 1M x 3'
for name,case in j['cases'].items():
 for ax,width in [(axes[0,0],.25),(axes[0,1],10)]:
  rows=[a for a in case['rows'] if a['window_width_mm']==width]
  ax.errorbar([a['depth_mm'] for a in rows],[100*a['valley']['relative_difference'] for a in rows],yerr=[200*a['valley']['gpu_only_ratio_se'] for a in rows],fmt='o-',c=colors[name],label=labels[name],capsize=3)
for ax,width in [(axes[0,0],.25),(axes[0,1],10)]:
 ax.axhline(0,c='#555',lw=.7);ax.set(xlabel='Depth (mm)',ylabel='Central valley GPU/TOPAS - 1 (%)',title=f'Depth window: {width:g} mm')
axes[0,0].legend(fontsize=8)
for a,label,c in [(t,'TOPAS 10M x 1','#111827')]+[(v.mean(0),labels[k],colors[k]) for k,v in data.items()]:
 mask=abs(z-100)<5
 axes[1,0].plot(x,a[mask].mean(0),c=c,label=label)
 axes[1,1].plot(x,a[mask].mean(0),c=c,label=label)
axes[1,0].set(xlabel='Transverse x (mm)',ylabel='Dose (Gy/source)',title='95-105 mm mean profile',xlim=(-5,5))
axes[1,1].set(xlabel='Transverse x (mm)',ylabel='Dose (Gy/source)',title='Right valley: 95-105 mm',xlim=(1.3,2.3))
axes[1,0].legend(fontsize=7,loc='upper right')
zoom=(x>=1.3)&(x<=2.3);depthmask=abs(z-100)<5
zoom_max=max(float(t[depthmask][:,zoom].mean(0).max()),*[float(v[:,depthmask][:,:,zoom].mean((0,1)).max()) for v in data.values()])
axes[1,1].set_ylim(0,zoom_max*1.12)
fig.suptitle('C12 250 MeV/u, EM only, Water_75eV | Only copper step ceiling changes in paired test\nError bars: +/-2 GPU run SE; TOPAS reference SE unavailable.',fontsize=12)
fig.savefig(o/'valley_diagnosis.png',dpi=170);plt.close(fig)
