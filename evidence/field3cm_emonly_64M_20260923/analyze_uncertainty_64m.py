from pathlib import Path
import csv,json,datetime,hashlib,os
import numpy as np
from scipy.stats import t as student
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=Path('/mnt/sdb/wuwei/MAIGO_review_fix_20260923')
O=Path(os.environ.get('MAIGO_ANALYSIS_INPUT',str(R/'evidence/field3cm_emonly_64M_20260923')));D=Path(os.environ.get('MAIGO_ANALYSIS_OUTPUT',str(O/'results')))
M=json.loads((O/'manifest.json').read_text());P=json.loads((O/'preflight.json').read_text())
C=json.loads((D/'comparison.json').read_text());S=json.loads((O/'serial_status.json').read_text())
assert C['status']=='COMPLETE' and S['status'] in ['COMPLETE','ANALYZING'];B=M['batches']
g=[];t=[];files=[]
for i in range(1,B+1):
    td=Path(M['cases'][f'topas_b{i}']);gd=Path(M['cases'][f'gpu_b{i}']).parent
    tq=json.loads((td/'quality.json').read_text());gq=json.loads((gd/'quality.json').read_text())
    assert tq['accepted'] and gq['accepted'] and tq['histories']==gq['histories']==M['histories_per_batch']
    assert tq['topas_binary_sha256']==P['topas_binary_sha256'] and gq['gpu_binary_sha256']==M['gpu_binary_sha256']
    for p,kind in [(td/'dose.bin','TOPAS'),(Path(gq['dose_path']),'GPU')]:
        b=p.read_bytes();files.append(dict(batch=i,engine=kind,path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
        a=np.frombuffer(b,dtype='<f8' if kind=='TOPAS' else '<f4').reshape(1000,1000).astype(float)/M['histories_per_batch']
        (t if kind=='TOPAS' else g).append(a)
g=np.stack(g);t=np.stack(t)
x=-49.95+np.arange(1000)*.1;z=.125+np.arange(1000)*.25
masks={'peak':abs(x)<.25,'valley':abs(abs(x)-1.8)<.45,
       'inner_field_peaks':np.any(abs(x[None,:]-np.arange(-3,4)[:,None]*3.6)<.25,axis=0),
       'inner_field_valleys':np.any(abs(x[None,:]-(np.arange(-3,3)+.5)[:,None]*3.6)<.45,axis=0)}

def stats(a,b):
    am=float(a.mean());bm=float(b.mean());r=am/bm
    v1=float(a.var(ddof=1)/B/bm**2);v2=float(b.var(ddof=1)/B*r*r/bm**2)
    se=np.sqrt(v1+v2);df=(v1+v2)**2/(v1*v1/(B-1)+v2*v2/(B-1))
    half=float(student.ppf(.975,df)*se*100)
    return dict(difference_percent=(r-1)*100,CI95_low_percent=(r-1)*100-half,
                CI95_high_percent=(r-1)*100+half,CI95_half_width_pp=half,
                difference_SE_pp=float(se*100),welch_df=df,
                gpu_batch_values=a.tolist(),topas_batch_values=b.tolist())

windows=[]
for lo,hi,label in [(d-5,d+5,f'{d} mm center, 10 mm window') for d in [20,40,60,80,100,120]]+[(20,120,'20-120 mm depth average')]:
    use=(z>=lo)&(z<hi)
    rows={key:stats(g[:,use][:,:,mask].mean((1,2)),t[:,use][:,:,mask].mean((1,2))) for key,mask in masks.items()}
    windows.append(dict(label=label,depth_lower_mm=lo,depth_upper_mm=hi,depth_bins=int(use.sum()),rois=rows))

rows=list(csv.DictReader((D/'depth_curves.csv').open()))
curves={}
for key in masks:
    rr=[r for r in rows if r['ROI']==key]
    assert len(rr)==1000
    curves[key]={c:np.array([float(r[c]) for r in rr]) for c in rr[0] if c!='ROI'}
    for item in C['rows']:
        k=int(np.argmin(abs(z-item['depth_mm'])))
        check=stats(g[:,k,masks[key]].mean(1),t[:,k,masks[key]].mean(1))
        assert abs(check['difference_percent']-item[key]['difference_percent'])<1e-9

def r80(a):
    i=int(np.argmax(a));v=.8*a[i];k=i+int(np.flatnonzero(a[i:]<=v)[0])
    return float(z[k-1]+(v-a[k-1])*(z[k]-z[k-1])/(a[k]-a[k-1]))

allvolume=stats(g.sum((1,2)),t.sum((1,2)))
assert abs(allvolume['difference_percent']-C['total_deposited_energy_difference_percent'])<1e-9
summary=dict(status='COMPLETE',method='Independent batches per engine (count from manifest); delta-method Welch-t 95% intervals. Each depth/space ROI is reduced inside each batch first, preserving correlations between depth voxels. Extra windows do not replace original 0.25 mm depth results.',
    whole_volume=allvolume,depth_windows=windows,
    entrance_first_layer={key:{c:float(curves[key][c][0]) for c in ['depth_mm','difference_percent','CI95_low_percent','CI95_high_percent']} for key in masks},
    integrated_depth_dose_R80_mm=dict(gpu=r80(g.mean(0).sum(1)),topas=r80(t.mean(0).sum(1))),
    wall_minutes=sum(v['wall_s'] for v in S['completed_stages'])/60,
    simulation_wall_s={engine:sum(v['wall_s'] for v in S['completed_stages'] if v['name'].startswith(engine+'_b')) for engine in ['topas','gpu']},
    peak_memory_bytes=S['peak_memory_bytes'],dose_files=files)
(D/'uncertainty_details.json').write_text(json.dumps(summary,indent=2)+'\n')

plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.2})
fig,axs=plt.subplots(2,2,figsize=(14,8),constrained_layout=True)
for col,(key,title) in enumerate([('inner_field_peaks','7 inner peaks combined'),('inner_field_valleys','6 inner valleys combined')]):
    c=curves[key];ax=axs[0,col]
    ax.plot(z,c['topas']*1e9,label='TOPAS',color='#2563eb')
    ax.plot(z,c['gpu']*1e9,label='GPU',color='#ea7b24',ls='--')
    ax.set(title=title,xlim=(0,140),ylim=(0,None),xlabel='Water depth (mm)',ylabel='Mean dose (nGy / source)');ax.legend()
    ax=axs[1,col];valid=(z<=125)&(c['topas']>.01*c['topas'].max())
    ax.axhspan(-1,1,color='#d7efda');ax.axhline(0,color='gray',lw=.6)
    ax.plot(z[valid],c['difference_percent'][valid],color='#007d83',lw=.85)
    ax.fill_between(z[valid],c['CI95_low_percent'][valid],c['CI95_high_percent'][valid],color='#007d83',alpha=.15)
    ax.set(xlim=(0,125),xlabel='Water depth (mm)',ylabel='(GPU/TOPAS - 1) (%)',title='Relative difference; approximate 95% CI')
fig.suptitle(f'3 cm x 3 cm field | Inner-field pooled peak/valley ROIs | {M["histories_per_engine"]/1e6:g}M histories per engine\n0.25 mm depth layers; 100 mm vertical average; no fitted normalization or depth smoothing')
fig.savefig(D/'inner_field_peak_valley_depth.png',dpi=180);plt.close(fig)
print(json.dumps({k:v for k,v in summary.items() if k not in ['depth_windows','dose_files']},indent=2))
for w in windows:print(w['label'],json.dumps({k:{c:v[c] for c in ['difference_percent','CI95_low_percent','CI95_high_percent']} for k,v in w['rois'].items()}))
