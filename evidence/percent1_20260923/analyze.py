from pathlib import Path
import numpy as np,json,csv,math,importlib.util
from scipy.stats import t as student
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=Path('/mnt/sdb/wuwei/MAIGO_review_fix_20260923');o=r/'evidence/percent1_20260923'
old=Path('/mnt/sda/wuwei/minibeam_single_center_em_only_water75ev_e250_10m/topas_7378')
gfiles=[r/f'out/valley_cu0051m_s{i}/dose.raw' for i in [1,2,3]]
completed=json.loads((o/'gpu_quality.json').read_text())['runs']
gfiles += [r/f'out/percent1_matched_s{i}/dose.raw' for i in range(1,13) if f'percent1_matched_s{i}' in completed]
g=np.stack([np.fromfile(f,dtype='<f4').reshape(1000,1000).astype(float)/1e6 for f in gfiles]);ng=np.full(len(g),1e6)
tdirs=[old]+[d for d in sorted(o.glob('topas_2m_s*')) if (d/'quality.json').exists()];nt=np.array([1e7]+[2e6]*(len(tdirs)-1))
t=np.stack([np.fromfile(d/'dose.bin',dtype='<f8').reshape(1000,1000)/n for d,n in zip(tdirs,nt)])
x=-49.95+np.arange(1000)*.1;z=.125+np.arange(1000)*.25;pm=abs(x)<.25;vm=abs(abs(x)-1.8)<.45
spec=importlib.util.spec_from_file_location('gate',r/'evidence/urban_d51e599_20260923/dose_gate.py');gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
def ws(values,n):
 values=np.asarray(values);mu=float(np.average(values,weights=n));k=len(n);N=float(sum(n))
 c=float(sum(n*(values-mu)**2)/(k-1)) if k>1 else None
 return {'mean':mu,'SE':math.sqrt(c/N) if c is not None else None,'variance_coefficient':c,'batches':k,'histories':int(N),'values':values.tolist(),'batch_histories':n.astype(int).tolist()}
def ratio(vg,vt):
 a,b=ws(vg,ng),ws(vt,nt);m=a['mean']/b['mean'];d=m-1
 ga=(a['SE']/b['mean'])**2;tb=(m*b['SE']/b['mean'])**2 if b['SE'] is not None else 0
 se=math.sqrt(ga+tb);df=(ga+tb)**2/(ga**2/(len(ng)-1)+(tb**2/(len(nt)-1) if len(nt)>1 else 0));crit=float(student.ppf(.975,df));crit6=float(student.ppf(1-.05/12,df))
 return {'gpu':a,'topas':b,'relative_difference':d,'combined_SE':se if len(nt)>1 else None,'gpu_only_SE':math.sqrt(ga),'topas_SE_contribution':math.sqrt(tb) if len(nt)>1 else None,'welch_df_approx':df if len(nt)>1 else None,'CI95_approx':[d-crit*se,d+crit*se] if len(nt)>1 else None,'zero_bias_p_approx':float(2*student.sf(abs(d)/se,df)) if len(nt)>1 else None,'passes_pointwise_1pct_95CI':abs(d)+crit*se<.01 if len(nt)>1 else False,'passes_six_depth_bonferroni_1pct':abs(d)+crit6*se<.01 if len(nt)>1 else False,'normal_95_halfwidth_1pct_history_multiplier':(1.96*se/.01)**2 if len(nt)>1 else None}
j={'status':'COMPLETE' if len(g)==15 and len(t)==4 else 'PARTIAL','method':'Fixed ROI, absolute per-source dose, weighted means. Var(mean)=sum(N_i*(x_i-weighted_mean)^2)/((k-1)*sum(N_i)); independent-batch covariance retained within each ROI; delta-method ratios, approximate Welch-t intervals. Existing TOPAS 10M + three new 2M batches; GPU 15 x 1M. Multiple-depth certification uses Bonferroni. Ratios/CI do not address physical model bias.','gpu_files':[str(f) for f in gfiles],'topas_directories':[str(d) for d in tdirs],'rows':[]}
for dep in [20,40,60,80,100,120]:
 for width in [.25,10]:
  mask=np.arange(1000)==int(np.argmin(abs(z-dep))) if width==.25 else abs(z-dep)<width/2
  row={'depth_mm':dep,'actual_depth_mm':float(z[mask].mean()),'window_width_mm':width}
  for label,m in [('valley',vm),('peak',pm),('left_valley',vm&(x<0)),('right_valley',vm&(x>0)),('row_mean',np.ones(1000,bool))]:
   row[label]=ratio(g[:,mask][:,:,m].mean((1,2)),t[:,mask][:,:,m].mean((1,2)))
  row['fwhm']=ratio([gate._fwhm_1d(v[mask].mean(0),x) for v in g],[gate._fwhm_1d(v[mask].mean(0),x) for v in t])
  if len(tdirs)>1:
   tn=t[1:,mask][:,:,vm].mean((1,2));to=float(t[0,mask][:,vm].mean());sn=ws(tn,nt[1:]);sec=math.sqrt(sn['variance_coefficient']*(1/nt[0]+1/sum(nt[1:])))
   row['new_topas_vs_old']={'relative_difference':sn['mean']/to-1,'SE_relative':sec/to,'new':sn,'old':to}
   carriers=[]
   for d in tdirs[1:]:
    e=np.fromfile(d/'dose_electron_carrier.bin',dtype='<f8').reshape(1000,1000)/2e6
    ne=np.fromfile(d/'dose_non_electron_carrier.bin',dtype='<f8').reshape(1000,1000)/2e6
    carriers.append({'electron_valley':float(e[mask][:,vm].mean()),'non_electron_valley':float(ne[mask][:,vm].mean()),'electron_peak':float(e[mask][:,pm].mean()),'non_electron_peak':float(ne[mask][:,pm].mean())})
   row['topas_carriers']=carriers
  j['rows'].append(row)
j['energy']=ratio(g.sum((1,2))*2.5e-6/1.602176634e-13,t.sum((1,2))*2.5e-6/1.602176634e-13)
j['r80']={'gpu':ws([gate._r80(v.sum(1),z) for v in g],ng),'topas':ws([gate._r80(v.sum(1),z) for v in t],nt)}
(o/'comparison.json').write_text(json.dumps(j,indent=2)+'\n')
print(j['status'],'GPU batches',len(g),'TOPAS histories',int(sum(nt)),flush=True)
for a in j['rows']:
 v=a['valley'];print(a['depth_mm'],a['window_width_mm'],'VALLEY%',round(100*v['relative_difference'],3),'GPU_SEpp',round(100*v['gpu_only_SE'],3),'COMBINED_SEpp',None if v['combined_SE'] is None else round(100*v['combined_SE'],3),'CI95%',None if v['CI95_approx'] is None else [round(100*z,3) for z in v['CI95_approx']],flush=True)
print('Energy%',100*j['energy']['relative_difference'],'R80diff',j['r80']['gpu']['mean']-j['r80']['topas']['mean'],flush=True)
with (o/'comparison.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['depth_mm','window_mm','valley_percent','GPU_SE_pp','combined_SE_pp','CI95_low_percent','CI95_high_percent','peak_percent','row_percent'])
 for a in j['rows']:
  v=a['valley'];ci=v['CI95_approx'];w.writerow([a['depth_mm'],a['window_width_mm'],100*v['relative_difference'],100*v['gpu_only_SE'],None if v['combined_SE'] is None else 100*v['combined_SE'],None if ci is None else 100*ci[0],None if ci is None else 100*ci[1],100*a['peak']['relative_difference'],100*a['row_mean']['relative_difference']])
fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
for ax,width in zip(axes,[.25,10]):
 rows=[a for a in j['rows'] if a['window_width_mm']==width];ds=[a['depth_mm'] for a in rows];vs=[100*a['valley']['relative_difference'] for a in rows]
 errs=[100*(a['valley']['CI95_approx'][1]-a['valley']['relative_difference']) if a['valley']['CI95_approx'] else 200*a['valley']['gpu_only_SE'] for a in rows]
 ax.axhspan(-1,1,color='#dcfce7');ax.axhline(0,color='#6b7280',lw=.7);ax.errorbar(ds,vs,yerr=errs,fmt='o-',color='#2563eb',capsize=4)
 ax.set(xlabel='Depth (mm)',ylabel='GPU/TOPAS - 1 (%)',title=f'Fixed central valleys; depth window {width:g} mm')
fig.suptitle(f"Matched Cu step 0.05 mm | GPU {len(g)}M / TOPAS {sum(nt)/1e6:g}M\n"+('Approximate 95% independent-batch intervals; green target +/-1%' if len(nt)>1 else '+/-2 GPU-only SE; TOPAS uncertainty pending'))
fig.savefig(o/'statistics.png',dpi=170);plt.close(fig)
