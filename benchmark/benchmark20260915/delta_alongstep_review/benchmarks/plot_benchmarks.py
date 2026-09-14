"""Requested b*-prefixed plots from completed, validated 3D benchmark pairs."""
from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
BASE=Path('/mnt/sdb/wuwei/MAIGO/benchmark/benchmark20260914/production_continuation')
OUT=R/'plots';OUT.mkdir(exist_ok=True)
def save(fig,name):
 fig.tight_layout()
 for ext in ('png','pdf'):fig.savefig(OUT/(name+'.'+ext),dpi=170)
 plt.close(fig)
def relative(g,t):
 return np.divide(100*(g-t),t,out=np.full_like(t,np.nan,dtype=float),where=np.isfinite(t)&(t!=0))
def plot(name):
 d=R/name;m=json.loads((d/'comparison.json').read_text());assert json.loads((d/'topas_status.json').read_text())['complete'];q=json.loads((d/'gpu_status.json').read_text());assert q['complete'] and q['quality']['queue_overflow_count']==0
 z,t,g=np.loadtxt(d/'idd.csv',delimiter=',').T;zb,tb,gb=np.loadtxt(BASE/name/'idd.csv',delimiter=',').T;assert np.array_equal(z,zb);peak=z[t.argmax()];limit=min(z[-1]+.25,1.2*peak)
 fig,ax=plt.subplots(figsize=(9,5));ax.plot(z,t,label='TOPAS');ax.plot(z,gb,label='GPU production',linestyle='--');ax.plot(z,g,label='GPU along-step');ax.set(xlim=(0,limit),xlabel='Depth (mm)',ylabel='MeV / primary / 0.5 mm',title=name+' IDD');ax.legend();ax.grid(alpha=.25);save(fig,name+'_idd')
 fig,ax=plt.subplots(2,1,figsize=(9,7),sharex=True)
 ax[0].plot(z,relative(gb,t),label='GPU production',linestyle='--');ax[0].plot(z,relative(g,t),label='GPU along-step');ax[0].legend();ax[0].set_ylabel('(GPU - TOPAS) / TOPAS (%)');ax[1].plot(z,100*(gb-t)/t.max(),linestyle='--');ax[1].plot(z,100*(g-t)/t.max());ax[1].set_ylabel('Difference / TOPAS peak (%)');ax[1].set_xlabel('Depth (mm)')
 for a in ax:a.set_xlim(0,limit);a.set_ylim(-5,5);a.axhline(0,color='gray',lw=.7);a.grid(alpha=.25)
 ax[0].set_title(name+' IDD relative error');save(fig,name+'_idd_relative_error')
 if not name.startswith('b4'):
  fits=json.loads((d/'sigma_fits.json').read_text());basefits=json.loads((BASE/name/'sigma_fits.json').read_text());fig,ax=plt.subplots(1,2,figsize=(12,4));ef,ea=plt.subplots(1,2,figsize=(12,4));cols=[]
  for i,key in enumerate(('core','halo')):
   values={}
   for code in ('topas','gpu'):
    series=fits[code];zz=np.array([v['depth_mm'] for v in series]);values[code]=np.array([v['fit'][key] if v['fit'] and v['fit']['valid'] else np.nan for v in series]);ax[i].plot(zz,values[code],label='TOPAS' if code=='topas' else 'GPU along-step')
   bv=np.array([v['fit'][key] if v['fit'] and v['fit']['valid'] else np.nan for v in basefits['gpu']]);ax[i].plot(zz,bv,label='GPU production',linestyle='--');ea[i].plot(zz,relative(bv,values['topas']),label='GPU production',linestyle='--');
   error=relative(values['gpu'],values['topas']);ea[i].plot(zz,error,label="GPU along-step");ea[i].legend();cols.extend([values['topas'],values['gpu'],error])
   ax[i].set(ylabel='Sigma (mm)',title=name+' '+key);ax[i].legend();ea[i].set(ylabel='(GPU - TOPAS) / TOPAS (%)',title=name+' '+key+' relative error');ea[i].axhline(0,color='gray',lw=.7)
   for a in (ax[i],ea[i]):a.set_xlim(0,limit);a.set_xlabel('Depth (mm)');a.grid(alpha=.25)
  save(fig,name+'_sigma_core_halo');save(ef,name+'_sigma_relative_error')
  np.savetxt(OUT/(name+'_sigma.csv'),np.column_stack([zz,*cols]),delimiter=',',header='depth_mm,topas_core,gpu_core,core_relative_pct,topas_halo,gpu_halo,halo_relative_pct')
 geom=json.loads((d/'geometry.json').read_text());shape=tuple(geom['shape']);top=np.zeros(shape);rho=np.zeros(shape)
 for j,b in enumerate(geom['blocks']):
  sl=np.s_[b['z0']:b['z1'],:,b['x0']:b['x1']];top[sl]=np.fromfile(d/f'dose{j}.bin','<f8').reshape(b['z1']-b['z0'],160,b['x1']-b['x0']);rho[sl]=b['rho']
 gpu=np.fromfile(d/'dose_gpu.raw','<f4').reshape(shape);factor=rho*.5e-6*6.241509074e12/geom['histories'];top*=factor;gpu=gpu*factor;baseline=np.fromfile(BASE/name/'dose_gpu.raw','<f4').reshape(shape)*factor;x=np.arange(160)-79.5
 # Nine nonidentical 2mm slabs at fixed fractions of the reference Bragg-peak depth.
 starts=[max(0,min(shape[0]-4,int(round((f*peak-1)/.5)))) for f in (.1,.3,.5,.7,.9,1.,1.05,1.1,1.18)]
 assert len(set(starts))==9
 profiles=[]
 for log in (False,True):
  fig,ax=plt.subplots(3,3,figsize=(14,11))
  for a,k in zip(ax.ravel(),starts):
   dep=(k+2)*.5
   for code,arr in [('TOPAS',top),('GPU production',baseline),('GPU along-step',gpu)]:
    v=arr[k:k+4].sum((0,1));a.plot(x,np.where(v>0,v,np.nan) if log else v,label=code)
    if not log:profiles.extend((dep,float(xx),code,float(yy)) for xx,yy in zip(x,v))
   if log:a.set_yscale('log')
   a.set(title=f'z = {dep:.2f} mm; 2 mm slab',xlabel='x (mm)',ylabel='MeV / primary / 1 mm x bin');a.grid(alpha=.2)
  ax[0,0].legend();fig.suptitle(name+' lateral profiles'+(' (log y)' if log else ' (linear y)'));save(fig,name+'_profiles_'+('log' if log else 'linear'))
 with (OUT/(name+'_profiles.csv')).open('w') as f:w=csv.writer(f);w.writerow(['depth_mm','x_mm','code','MeV_per_primary']);w.writerows(profiles)
 return dict(case=name,depth_limit_mm=float(limit),profile_depths_mm=[(k+2)*.5 for k in starts],straggling_scale=next(line.split(':',1)[1].strip() for line in (d/'gpu.yaml').read_text().splitlines() if line.startswith('straggling_scale:')))
def main(selected=None):
 rows=[]
 existing={v["case"]:v for v in json.loads((OUT/"plot_manifest.json").read_text())} if (OUT/"plot_manifest.json").exists() else {}
 for c in json.loads((R/'cases.json').read_text()):
  if selected is not None and c[0] not in selected:rows.append(existing[c[0]]);continue
  if (R/c[0]/'comparison.json').exists():rows.append(plot(c[0]));print(c[0],'plots ready',flush=True)
 (OUT/'plot_manifest.json').write_text(json.dumps(rows,indent=2))
 text=['# Benchmark plots','','文件名均以 case ID 开头。IDD 深度范围为 TOPAS 最大 IDD 所在深度 ×1.2，受体模范围限制。三维 scorer 横向求和，没有使用 1D scorer。','','b1–b3：idd、idd_relative_error、sigma_core_halo、sigma_relative_error、profiles_linear、profiles_log。b4：idd、idd_relative_error、profiles_linear、profiles_log。各图提供 PNG/PDF。','','Relative error = 100 × (GPU−TOPAS)/TOPAS；IDD 另画以 TOPAS 峰值归一的差值，两种 IDD 误差图纵轴均限制为 −5% 至 +5%，原始数据不裁剪。sigma 仅在双方拟合均有效时计算误差，缺失处不表示零误差。log profile 的零值不显示。横向 profile 对 y 和精确 2 mm 深度窗积分，单位 MeV/primary/x-bin；九个深度由参考峰位的固定比例选取，实际深度见图和 manifest。','','| Case | GPU straggling scale | IDD | Profiles |','|---|---:|---|---|']
 for row in rows:
  n=row['case'];text.append(f'| {n} | {row["straggling_scale"]} | [{n}_idd]({n}_idd.png) | [linear]({n}_profiles_linear.png) / [log]({n}_profiles_log.png) |')
 (OUT/'README.md').write_text('\n'.join(text)+'\n')
if __name__=='__main__':main()
