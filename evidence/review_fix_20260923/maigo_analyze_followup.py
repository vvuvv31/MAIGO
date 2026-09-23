from pathlib import Path
import json, numpy as np, importlib.util, hashlib, re, csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('/mnt/sdb/wuwei/MAIGO_review_fix_20260923')
out=root/'evidence/review_fix_20260923'
ref=Path('/mnt/sda/wuwei/minibeam_single_center_em_only_water75ev_e250_10m/topas_7378')
spec=importlib.util.spec_from_file_location('gate',root/'evidence/urban_d51e599_20260923/dose_gate.py')
gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(a):
 a=np.asarray(a,float)
 return {'mean':float(a.mean()),'se':float(a.std(ddof=1)/np.sqrt(len(a))) if len(a)>1 else None,'values':a.tolist()}
def ratio(a,b):
 aa,bb=stats(a),stats(b)
 return {'gpu':aa,'topas':bb,'relative_difference':aa['mean']/bb['mean']-1 if bb['mean'] else None,'combined_ratio_se':None}
x=-49.95+np.arange(1000)*.1;z=.125+np.arange(1000)*.25
mass=2.5e-6;mevj=1.602176634e-13
ref_log=ref.with_suffix('.log').read_text()
assert 'CarbonPBS: Total number of histories: 10000000' in ref_log
assert ref.joinpath('dose.bin').stat().st_size==1000*1000*8
g=np.stack([np.fromfile(root/f'out/review_fullchain_s{i}/dose.raw',dtype='<f4').reshape(1000,1000).astype(float)/250000 for i in [1,2,3]])
t=np.fromfile(ref/'dose.bin',dtype='<f8').reshape(1,1000,1000)/10000000
r={'scope':'Cu collimator -> 60 mm air gap -> Water_75eV, carbon EM only, 250 MeV/u, central TPS spot; same configured distributions, independent source samples',
   'status':'EXPLORATORY_REFERENCE_SE_MISSING','gpu_histories_per_seed':250000,'gpu_seeds':[2026092601,2026092602,2026092603],
   'topas_histories':10000000,'topas_independent_runs':1,'reference':str(ref),
   'normalization':'Gy per source primary; no fitted normalization','energy_MeV_per_primary':ratio(g.sum((1,2))*mass/mevj,t.sum((1,2))*mass/mevj),'rows':[]}
pm=abs(x)<.25;vm=abs(abs(x)-1.8)<.45
for dep in [20,40,60,80,100,120]:
 iz=np.argmin(abs(z-dep));wg=[gate._fwhm_1d(a[iz],x) for a in g];wt=[gate._fwhm_1d(a[iz],x) for a in t]
 pg,pt=g[:,iz,pm].mean(1),t[:,iz,pm].mean(1);vg,vt=g[:,iz,vm].mean(1),t[:,iz,vm].mean(1)
 r['rows'].append({'depth_mm':float(z[iz]),'row_energy':ratio(g[:,iz].sum(1),t[:,iz].sum(1)),
 'peak':ratio(pg,pt),'valley':ratio(vg,vt),'pvdr':ratio(pg/vg,pt/vt),
 'fwhm_mm':ratio(wg,wt) if None not in wg+wt else None})
rg=[gate._r80(a.sum(1),z) for a in g];rt=[gate._r80(a.sum(1),z) for a in t]
r['integrated_depth_R80_mm']={'gpu':stats(rg),'topas':stats(rt),'difference_mm':float(np.mean(rg)-np.mean(rt))}
r['provenance']={str(p):sha(p) for p in [ref/'dose.bin',ref/'dose.binheader',ref/'run_single_center_em_only_water75ev_e250_10m.txt',ref.with_suffix('.log')]+[root/f'config/review_fullchain_s{i}.yaml' for i in [1,2,3]]}
(out/'fullchain_comparison.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
with (out/'fullchain_comparison.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['depth_mm','row_energy_relative','peak_relative','valley_relative','PVDR_relative','GPU_FWHM_mm','TOPAS_FWHM_mm','FWHM_relative'])
 for row in r['rows']:
  fw=row['fwhm_mm'];w.writerow([row['depth_mm']]+[row[k]['relative_difference'] for k in ['row_energy','peak','valley','pvdr']]+[fw['gpu']['mean'],fw['topas']['mean'],fw['relative_difference']])
fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
for a,label,c in [(t,'TOPAS: 10M x 1','#111827'),(g,'GPU fixed: 250k x 3','#dc2626')]:
 axes[0,0].plot(z,a.sum(2).mean(0)*mass/mevj/.25,label=label,c=c)
axes[0,0].set(xlabel='Depth (mm)',ylabel='Deposited energy (MeV/mm/source primary)',xlim=(0,145));axes[0,0].legend()
axes[0,1].plot(z,g.sum(2).mean(0)/np.maximum(t.sum(2).mean(0),1e-30)-1,c='#dc2626');axes[0,1].axhline(0,c='#888',lw=.7)
axes[0,1].set(xlabel='Depth (mm)',ylabel='GPU/TOPAS - 1',xlim=(0,120),ylim=(-.15,.15))
for role,label,c in [('gpu','GPU fixed','#dc2626'),('topas','TOPAS','#111827')]:
 axes[0,2].errorbar([a['depth_mm'] for a in r['rows']],[a['fwhm_mm'][role]['mean'] for a in r['rows']],
 yerr=[2*(a['fwhm_mm'][role]['se'] or 0) for a in r['rows']],fmt='o-',label=label,c=c,capsize=3)
axes[0,2].set(xlabel='Depth (mm)',ylabel='Central beamlet FWHM (mm)');axes[0,2].legend()
for ax,dep in zip(axes[1],[20,80,120]):
 iz=np.argmin(abs(z-dep))
 for a,label,c in [(t,'TOPAS','#111827'),(g,'GPU fixed','#dc2626')]:ax.plot(x,a[:,iz].mean(0),c=c,label=label)
 ax.set(xlabel='Transverse x (mm)',ylabel='Dose (Gy/source primary)',xlim=(-8,8),title=f'Depth {z[iz]:.3f} mm')
fig.suptitle('C12 250 MeV/u | Cu collimator to water | EM only | No dose renormalization\nExploratory: TOPAS reference has one seed, so its run SE is unavailable. GPU error bars: +/-2 SE.',fontsize=12)
fig.savefig(out/'topas_gpu_fullchain_comparison.png',dpi=180)
plt.close(fig)
follow={'scope':'Small smoke checks; 2000 histories at each step ceiling is not a precision convergence certificate','runs':{},'step_comparison':{}}
for name in ['terminal','step050','step025','fullchain_s1','fullchain_s2','fullchain_s3']:
 d=root/f'out/review_{name}';q=json.loads((d/'quality_report.json').read_text());e=json.loads((d/'energy_ledger.json').read_text());log=(out/f'gpu_{name}.log').read_text()
 urban=json.loads(re.search(r'URBAN_RUN_QUALITY (\{[^\n]+\})',log).group(1))
 follow['runs'][name]={'histories':e['histories'],'quality_accepted':q['accepted'],'failures':q['failures'],
 'energy_residual_relative':q['relative_energy_residual'],'energy_MeV':{k:e[k] for k in ['E_in_MeV','E_dep_MeV','E_esc_MeV','E_beamline_MeV','E_untracked_MeV']},'urban':urban}
 assert q['accepted'] and not q['failures'] and not any(urban[k] for k in ['fatal','cap','guard','subulp'])
zs=.125+np.arange(600)*.25
for engine in ['gpu','topas']:
 rows=[]
 for label in ['050','025']:
  a=np.fromfile(root/f'out/review_step{label}/dose.raw',dtype='<f4').reshape(600,1000).astype(float)/2000 if engine=='gpu' else np.fromfile(out/f'topas_step{label}/dose.bin',dtype='<f8').reshape(600,1000)/2000
  rows.append({'step_mm':float(label)/1000,'R80_mm':gate._r80(a.sum(1),zs),'energy_MeV_per_primary':float(a.sum()*mass/mevj),
  'fwhm_mm':[gate._fwhm_1d(a[np.argmin(abs(zs-dep))],x) for dep in [20,80,120]]})
 follow['step_comparison'][engine]={'runs':rows,'R80_change_mm':rows[1]['R80_mm']-rows[0]['R80_mm'],
  'FWHM_change_relative':(np.array(rows[1]['fwhm_mm'])/rows[0]['fwhm_mm']-1).tolist()}
(out/'followup_checks.json').write_text(json.dumps(follow,indent=2,allow_nan=False)+'\n')
print('FULLCHAIN ENERGY',r['energy_MeV_per_primary']);print('FULLCHAIN R80',r['integrated_depth_R80_mm'])
print((out/'fullchain_comparison.csv').read_text());print('STEP',json.dumps(follow['step_comparison'],indent=2))
