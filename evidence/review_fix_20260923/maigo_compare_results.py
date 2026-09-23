from pathlib import Path
import numpy as np,json,hashlib,importlib.util,subprocess,csv,sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('/mnt/sdb/wuwei/MAIGO_review_fix_20260923');out=root/'evidence/review_fix_20260923'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
gate_path=root/'evidence/urban_d51e599_20260923/dose_gate.py'
spec=importlib.util.spec_from_file_location('dose_gate',gate_path);gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
manifest={role:{'schema':'maigo.dose-scorer-manifest.v1','role':role,'runs':[]} for role in ['gpu','topas']}
for i in [1,2,3]:
 for role in ['gpu','topas']:
  folder=root/f'out/review_water_fixed_s{i}' if role=='gpu' else out/f'topas_water_s{i}'
  cfg=root/f'config/review_water_fixed_s{i}.yaml' if role=='gpu' else folder/'input.txt'
  dose=folder/('dose.mhd' if role=='gpu' else 'dose.bin')
  r={'effective_config':str(cfg),'config_sha256':sha(cfg),'history_count':20000,
     'random_seed':2026092300+i if role=='gpu' else 2026092400+i,
     'normalization':'Gy_sum_over_histories','density_g_cm3':1.,'dose_sha256':sha(dose),
     'geometry_axes':{'transverse':'x','depth':'z','slab_thickness':'y'},
     'origin_centers_mm':[-49.95,.125],'shape_xz':[1000,600],
     'spacing_xz_mm':[.1,.25],'slab_thickness_mm':100.}
  if role=='gpu':r.update(mhd=str(dose),mhd_sha256=sha(dose),raw_sha256=sha(folder/'dose.raw'))
  else:r.update(dose_bin=str(dose),dose_header=str(folder/'dose.binheader'),header_sha256=sha(folder/'dose.binheader'),element_type='float64')
  manifest[role]['runs'].append(r)
for role,m in manifest.items():(out/f'{role}_manifest.json').write_text(json.dumps(m,indent=2)+'\n')
cmd=[sys.executable,str(gate_path),'--gpu-manifest',str(out/'gpu_manifest.json'),'--topas-manifest',str(out/'topas_manifest.json'),
     '--accept',str(root/'evidence/urban_after_0f2c0ca_20260922/acceptance.yaml'),'--out',str(out/'water_dose_gate.json')]
run=subprocess.run(cmd,capture_output=True,text=True)
(out/'water_dose_gate.log').write_text(run.stdout+run.stderr+f'\nexit_code={run.returncode}\n')
g=np.stack([np.fromfile(root/f'out/review_water_fixed_s{i}/dose.raw',dtype='<f4').reshape(600,1000).astype(float)/20000 for i in [1,2,3]])
t=np.stack([np.fromfile(out/f'topas_water_s{i}/dose.bin',dtype='<f8').reshape(600,1000)/20000 for i in [1,2,3]])
before=np.stack([np.fromfile(root/f'out/review_water_beamlet_s{i}/dose.raw',dtype='<f4').reshape(600,1000).astype(float)/20000 for i in [1,2,3]])
x=-49.95+np.arange(1000)*.1;z=.125+np.arange(600)*.25
mass=2.5e-6;mevJ=1.602176634e-13
def stats(a):
 a=np.asarray(a,float);return {'mean':float(a.mean()),'se':float(a.std(ddof=1)/np.sqrt(len(a))),'values':a.tolist()}
def ratio(a,b):
 aa,bb=stats(a),stats(b)
 if bb['mean']==0:return {'gpu':aa,'topas':bb,'relative_difference':None}
 r=aa['mean']/bb['mean'];se=np.hypot(aa['se']/bb['mean'],aa['mean']*bb['se']/bb['mean']**2)
 return {'gpu':aa,'topas':bb,'relative_difference':r-1,'ratio_se':float(se)}
result={'scope':'C12 250 MeV/u, 0.5 mm uniform beamlet, no Cu, Water_75eV, identical entrance records, 3 independent seeds per engine, 20000 histories/seed',
        'dose_gate_exit_code':run.returncode,'dose_gate':json.loads((out/'water_dose_gate.json').read_text()),
        'energy_deposited_MeV_per_primary':ratio(g.sum((1,2))*mass/mevJ,t.sum((1,2))*mass/mevJ),'rows':[]}
for dep in [20,40,60,80,100,120]:
 iz=np.argmin(abs(z-dep));pm=abs(x)<.25;vm=abs(abs(x)-1.8)<.45
 wg=[gate._fwhm_1d(a[iz],x) for a in g];wt=[gate._fwhm_1d(a[iz],x) for a in t]
 row={'depth_mm':float(z[iz]),'row_energy':ratio(g[:,iz].sum(1),t[:,iz].sum(1)),
      'peak':ratio(g[:,iz,pm].mean(1),t[:,iz,pm].mean(1)),
      'valley':ratio(g[:,iz,vm].mean(1),t[:,iz,vm].mean(1)),
      'fwhm_mm':ratio(wg,wt) if None not in wg+wt else None}
 result['rows'].append(row)
cx=np.argmin(abs(x))
for kind,axis in [('central_axis',False),('integrated_depth',True)]:
 rg=[gate._r80(a.sum(1) if axis else a[:,cx],z) for a in g]
 rt=[gate._r80(a.sum(1) if axis else a[:,cx],z) for a in t]
 result[kind+'_R80_mm']={'gpu':stats(rg),'topas':stats(rt),'difference_mm':float(np.mean(rg)-np.mean(rt))}
result['gpu_quality']=[json.loads((root/f'out/review_water_fixed_s{i}/quality_report.json').read_text()) for i in [1,2,3]]
(out/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
with (out/'comparison.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['depth_mm','row_energy_relative','peak_relative','valley_relative','GPU_FWHM_mm','TOPAS_FWHM_mm','FWHM_relative'])
 for r in result['rows']:
  fw=r['fwhm_mm'];w.writerow([r['depth_mm'],r['row_energy']['relative_difference'],r['peak']['relative_difference'],r['valley']['relative_difference'],fw['gpu']['mean'] if fw else '',fw['topas']['mean'] if fw else '',fw['relative_difference'] if fw else ''])
fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
for a,label,c in [(t,'TOPAS','#111827'),(g,'GPU fixed','#dc2626'),(before,'GPU previous draft','#9ca3af')]:
 idd=a.sum(2).mean(0)*mass/mevJ/.25
 axes[0,0].plot(z,idd,label=label,c=c,lw=1.4)
axes[0,0].set(xlabel='Depth in water (mm)',ylabel='Deposited energy (MeV/mm/primary)',xlim=(0,140));axes[0,0].legend()
axes[0,1].plot(z,g.sum(2).mean(0)/np.maximum(t.sum(2).mean(0),1e-30)-1,c='#dc2626');axes[0,1].axhline(0,c='#888',lw=.7)
axes[0,1].set(xlabel='Depth (mm)',ylabel='GPU/TOPAS - 1',xlim=(0,125),ylim=(-.08,.08))
valid=[r for r in result['rows'] if r['fwhm_mm']]
for role,label,c in [('gpu','GPU fixed','#dc2626'),('topas','TOPAS','#111827')]:
 axes[0,2].errorbar([r['depth_mm'] for r in valid],[r['fwhm_mm'][role]['mean'] for r in valid],yerr=[2*r['fwhm_mm'][role]['se'] for r in valid],fmt='o-',label=label,c=c,capsize=3)
axes[0,2].set(xlabel='Depth (mm)',ylabel='Transverse beamlet FWHM (mm)');axes[0,2].legend()
for ax,dep in zip(axes[1],[20,80,120]):
 iz=np.argmin(abs(z-dep))
 for a,label,c in [(t,'TOPAS','#111827'),(g,'GPU fixed','#dc2626')]:ax.plot(x,a[:,iz].mean(0),c=c,label=label)
 ax.set(xlabel='Transverse x (mm)',ylabel='Dose (Gy/primary)',xlim=(-4,4),title=f'Depth {z[iz]:.3f} mm')
fig.suptitle('C12 250 MeV/u | 0.5 mm beamlet | Water only | 20,000 histories × 3 seeds per engine\nSame input records; 0.05 mm step ceiling. Error bars: ±2 run SE; exploratory comparison.',fontsize=12)
fig.savefig(out/'topas_gpu_water_comparison.png',dpi=180)
print('ENERGY',result['energy_deposited_MeV_per_primary'])
print('R80',result['integrated_depth_R80_mm'])
print((out/'comparison.csv').read_text())
print(run.stdout,run.stderr)
