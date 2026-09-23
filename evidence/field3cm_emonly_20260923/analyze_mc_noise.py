from pathlib import Path
import csv, datetime, hashlib, json, re
import numpy as np
from scipy.stats import t as student, chi2
import pymedphys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

O=Path('/mnt/sdb/wuwei/MAIGO_review_fix_20260923/evidence/field3cm_emonly_20260923')
D=O/'results/mc_noise_12p8M';D.mkdir(exist_ok=True)
M=json.loads((O/'manifest.json').read_text())
U=json.loads((O/'results/uncertainty_details.json').read_text())
assert json.loads((O/'serial_status.json').read_text())['status']=='COMPLETE'
B=M['batches'];N=M['histories_per_engine'];assert B==4 and N==12800000
expected={v['path']:v['sha256'] for v in U['dose_files']}
data={'TOPAS':[],'GPU':[]};inputs=[];seeds={'TOPAS':[],'GPU':[]}
for i in range(1,5):
    td=Path(M['cases'][f'topas_b{i}']);gc=Path(M['cases'][f'gpu_b{i}'])
    tq=json.loads((td/'quality.json').read_text());gq=json.loads((gc.parent/'quality.json').read_text())
    assert tq['accepted'] and gq['accepted'] and tq['histories']==gq['histories']==M['histories_per_batch']
    seeds['TOPAS'].append(int(re.search(r'i:Ts/Seed\s*=\s*(\d+)',(td/'run.txt').read_text()).group(1)))
    seeds['GPU'].append(int(re.search(r'^random_seed:\s*(\d+)',gc.read_text(),re.M).group(1)))
    for engine,p,dt in [('TOPAS',td/'dose.bin','<f8'),('GPU',Path(gq['dose_path']),'<f4')]:
        raw=p.read_bytes();h=hashlib.sha256(raw).hexdigest();assert h==expected[str(p)]
        a=np.frombuffer(raw,dtype=dt).reshape(1000,1000).astype(float)/M['histories_per_batch']
        assert np.all(np.isfinite(a)) and np.all(a>=0)
        data[engine].append(a);inputs.append(dict(engine=engine,batch=i,path=str(p),sha256=h))
assert len(set(seeds['TOPAS']+seeds['GPU']))==8
data={k:np.stack(v) for k,v in data.items()}
mean={k:v.mean(0) for k,v in data.items()}
vm={k:v.var(axis=0,ddof=1)/B for k,v in data.items()}
t=mean['TOPAS'];g=mean['GPU'];vt=vm['TOPAS'];vg=vm['GPU']
x=-49.95+np.arange(1000)*.1;z=.125+np.arange(1000)*.25
mask=t>=.03*t.max();assert mask.sum()==201625
previous=np.load(O/'results/gamma_0p2mm_3pct_local_global_cutoff3/gamma_maps.npz')
assert np.array_equal(mask,previous['evaluated'])
rois={'central_peak':abs(x)<.25,'adjacent_valleys':abs(abs(x)-1.8)<.45,
      'inner_field_peaks':np.any(abs(x[None,:]-np.arange(-3,4)[:,None]*3.6)<.25,axis=0),
      'inner_field_valleys':np.any(abs(x[None,:]-(np.arange(-3,3)+.5)[:,None]*3.6)<.45,axis=0)}

def safe_ratio(a,b):
    return np.divide(a,b,out=np.full(np.broadcast_shapes(np.shape(a),np.shape(b)),np.nan),where=b>0)

rel_se={k:100*safe_ratio(np.sqrt(vm[k]),mean[k]) for k in data}
# Relative dose-difference uncertainty with the reference mean used as a fixed reporting scale.
diff_se=100*safe_ratio(np.sqrt(vg+vt),t)
observed=100*safe_ratio(g-t,t)

def distribution(a):
    a=np.asarray(a);a=a[np.isfinite(a)]
    return dict(n=int(a.size),mean=float(a.mean()),rms=float(np.sqrt(np.mean(a*a))),
                **{f'p{p}':float(np.percentile(a,p)) for p in [10,25,50,75,90,95]})

def voxel_stats(m):
    return dict(voxels=int(m.sum()),TOPAS_SE_percent=distribution(rel_se['TOPAS'][m]),
                GPU_SE_percent=distribution(rel_se['GPU'][m]),
                difference_SE_pp=distribution(diff_se[m]),observed_difference_percent=distribution(observed[m]),
                local_0mm_pass_percent=float(np.mean(abs(observed[m])<=3)*100),
                local_0p2mm_pass_percent=float(np.mean(previous['gamma_local'][m]<=1)*100),
                estimated_difference_SE_above_1pp_percent=float(np.mean(diff_se[m]>1)*100),
                estimated_difference_SE_above_3pp_percent=float(np.mean(diff_se[m]>3)*100))

groups={'all_gamma_voxels':voxel_stats(mask)}
mid=(z>=20)&(z<120)
for key,r in rois.items():groups[key+'_20_120mm_voxels']=voxel_stats(mask&mid[:,None]&r[None,:])
dose_bands=[]
for lo,hi in [(3,5),(5,10),(10,20),(20,50),(50,100.000001)]:
    m=mask&(t>=lo/100*t.max())&(t<hi/100*t.max())
    dose_bands.append(dict(lower_Dmax_percent=lo,upper_Dmax_percent=min(hi,100),**voxel_stats(m)))
print('Voxel uncertainty:',json.dumps({k:{c:v[c] for c in ['TOPAS_SE_percent','GPU_SE_percent','difference_SE_pp']} for k,v in groups.items()}),flush=True)

def aggregate_stats(a,b):
    # a = GPU batch ROI doses; b = TOPAS batch ROI doses; preserve spatial covariance.
    am=a.mean(axis=0);bm=b.mean(axis=0);va=a.var(axis=0,ddof=1)/B;vb=b.var(axis=0,ddof=1)/B
    ratio=safe_ratio(am,bm);q1=safe_ratio(va,bm**2);q2=safe_ratio(vb*ratio**2,bm**2)
    vv=q1+q2;df=safe_ratio(vv**2,(q1*q1+q2*q2)/(B-1))
    se=100*np.sqrt(vv);half=student.ppf(.975,df)*se
    return dict(TOPAS_mean=bm,GPU_mean=am,TOPAS_SE_percent=100*safe_ratio(np.sqrt(vb),bm),
                GPU_SE_percent=100*safe_ratio(np.sqrt(va),am),difference_percent=100*(ratio-1),
                ratio_SE_pp=se,CI95_low_percent=100*(ratio-1)-half,CI95_high_percent=100*(ratio-1)+half,
                CI95_half_width_pp=half,welch_df=df)

def scalar(d):return {k:float(v) for k,v in d.items()}
layers={};selected=[];layer_summaries={}
for key,r in rois.items():
    a=data['GPU'][:,:,r].mean(axis=2);b=data['TOPAS'][:,:,r].mean(axis=2)
    st=aggregate_stats(a,b);layers[key]=st
    layer_summaries[key]={k:distribution(v[mid]) for k,v in st.items() if k not in ['TOPAS_mean','GPU_mean']}
    for requested in [0,20,40,60,80,100,120]:
        k=int(np.argmin(abs(z-requested)))
        selected.append(dict(ROI=key,requested_depth_mm=requested,depth_mm=float(z[k]),
                             **{c:float(v[k]) for c,v in st.items()}))
windows={key:scalar(aggregate_stats(data['GPU'][:,mid][:,:,r].mean((1,2)),data['TOPAS'][:,mid][:,:,r].mean((1,2)))) for key,r in rois.items()}
whole=scalar(aggregate_stats(data['GPU'].sum((1,2)),data['TOPAS'].sum((1,2))))
assert abs(whole['difference_percent']-U['whole_volume']['difference_percent'])<1e-10
for key,st in windows.items():
    old=U['depth_windows'][-1]['rois'][{'central_peak':'peak','adjacent_valleys':'valley'}.get(key,key)]
    assert abs(st['ratio_SE_pp']-old['difference_SE_pp'])<1e-10
print('Layer ROI SE medians:',json.dumps({k:{c:v[c]['p50'] for c in ['TOPAS_SE_percent','GPU_SE_percent','ratio_SE_pp','CI95_half_width_pp']} for k,v in layer_summaries.items()}),flush=True)

# Same-engine comparisons directly expose random-seed variation without a transport-model difference.
# These have 6.4M histories on EACH side, so the difference noise is sqrt(2) larger than
# a same-engine 12.8M vs 12.8M comparison. Report raw results without correcting gamma rates.
self_checks=[]
for engine,a in data.items():
    for left,right in [([0,1],[2,3]),([0,2],[1,3]),([0,3],[1,2])]:
        ref=a[left].mean(0);eva=a[right].mean(0);m=ref>=.03*ref.max()
        result=dict(engine=engine,reference_batches=[i+1 for i in left],evaluation_batches=[i+1 for i in right],
                    histories_per_side=N//2,evaluated_voxels=int(m.sum()))
        for mode,local in [('global',False),('local',True)]:
            denom=.03*ref[m] if local else .03*ref.max()
            result[mode+'_0mm_pass_percent']=float(np.mean(abs(eva[m]-ref[m])/denom<=1)*100)
            gamma=pymedphys.gamma((z,x),ref,(z,x),eva,3.,.2,lower_percent_dose_cutoff=3.,
                interp_fraction=10,max_gamma=2.,local_gamma=local,global_normalisation=float(ref.max()),
                skip_once_passed=False,random_subset=None,ram_available=1024**3,interp_algo='scipy')
            assert np.array_equal(np.isfinite(gamma),m)
            result[mode+'_0p2mm_pass_percent']=float(np.mean(gamma[m]<=1)*100)
        self_checks.append(result);print('Self comparison:',json.dumps(result),flush=True)

planning={}
for key in ['inner_field_peaks_20_120mm_voxels','inner_field_valleys_20_120mm_voxels']:
    planning[key]={}
    for percentile in ['p50','p90']:
        s=groups[key]['difference_SE_pp'][percentile]
        planning[key][percentile]=dict(current_difference_SE_pp=s,
            histories_per_engine_for_1pp_SE=N*max(1,s*s),
            histories_per_engine_for_1pp_95_normal_halfwidth=N*max(1,(1.96*s)**2))
for key,st in layer_summaries.items():
    s=st['ratio_SE_pp']['p50']
    planning[key+'_layer_ROI_median']=dict(current_ratio_SE_pp=s,
        histories_per_engine_for_1pp_SE=N*max(1,s*s),
        histories_per_engine_for_1pp_95_normal_halfwidth=N*max(1,(1.96*s)**2))

cg=next(s.split('::',1)[1] for s in Path('/proc/self/cgroup').read_text().splitlines() if s.startswith('0::'))
cg=Path('/sys/fs/cgroup')/cg.lstrip('/')
assert int((cg/'memory.max').read_text())<=30000000000 and (cg/'memory.swap.max').read_text().strip()=='0'
summary=dict(status='COMPLETE',completed_at=datetime.datetime.now().astimezone().isoformat(),
    histories_per_engine=N,histories_per_batch=N//B,batches=B,random_seeds=seeds,
    method='Independent-seed batch variance; variance of final mean = unbiased batch variance / 4. All ROI doses aggregated within each batch before estimating variance.',
    voxel_difference_SE_definition='100 * sqrt(SE_GPU_absolute^2 + SE_TOPAS_absolute^2) / mean_TOPAS; reference mean is a reporting scale.',
    ROI_ratio_SE_definition='Delta-method SE of 100*(GPU/TOPAS-1), with both independent variances propagated; Welch-t 95% intervals.',
    voxel_mask='TOPAS four-batch mean >= 3% of its Dmax; 201625 voxels, identical to the latest full-dose gamma.',
    voxel_groups=groups,dose_bands=dose_bands,layer_ROI_distributions_20_120mm=layer_summaries,
    selected_depth_ROIs=selected,depth_20_120mm_averaged_ROIs=windows,whole_volume=whole,
    same_engine_half_sample_gamma=self_checks,
    same_engine_note='Three overlapping 2+2 partitions per engine, not three independent experiments. Each side 6.4M histories. Their difference SE is sqrt(2) higher than 12.8M-vs-12.8M within the same engine. Masks use each half reference and may differ from the full-dose mask. No pass-rate rescaling or confidence interval is inferred.',
    planning=planning,planning_note='Both engines scaled equally, independent histories, unchanged grid and physics, SE proportional to N^-1/2. 95% planning uses asymptotic normal 1.96 and many future independent batches; current four-batch CIs use Welch t. These are pointwise precision estimates, not a guarantee of <=1% bias or simultaneous bounds over all voxels.',
    single_engine_SE_95_variance_estimate_multiplier=[float(np.sqrt(3/chi2.ppf(.975,3))),float(np.sqrt(3/chi2.ppf(.025,3)))],
    limitations=['Only four batches: each voxel variance has three degrees of freedom and is imprecisely estimated.',
        'Spatially neighboring voxels and overlapping partitions are correlated; no binomial confidence interval or independent-voxel sample-size claim is made.',
        'Scoring uses 0.1 x 100 x 0.25 mm voxels with 100 mm vertical averaging; noise is resolution dependent.',
        'Observed RMS and estimated noise RMS being similar does not rule out structured bias. No new particle simulations were run.'],
    pymedphys_version=pymedphys.__version__,memory_max_bytes=int((cg/'memory.max').read_text()),
    memory_peak_bytes=int((cg/'memory.peak').read_text()),inputs=inputs)

with (D/'selected_depth_ROIs.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(selected[0]));w.writeheader();w.writerows(selected)
depth_rows=[]
for k,depth in enumerate(z):
    nn=int(mask[k].sum());row=dict(depth_mm=float(depth),evaluated_voxels=nn)
    row['voxel_difference_noise_RMS_pp']=float(np.sqrt(np.mean(diff_se[k,mask[k]]**2))) if nn else None
    row['observed_voxel_difference_RMS_percent']=float(np.sqrt(np.mean(observed[k,mask[k]]**2))) if nn else None
    for key,st in layers.items():
        for metric in ['TOPAS_SE_percent','GPU_SE_percent','ratio_SE_pp','CI95_half_width_pp']:row[key+'_'+metric]=float(st[metric][k])
    depth_rows.append(row)
with (D/'noise_by_depth.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(depth_rows[0]));w.writeheader();w.writerows(depth_rows)
np.savez_compressed(D/'noise_maps.npz',TOPAS_SE_percent=np.where(mask,rel_se['TOPAS'],np.nan).astype('f4'),
    GPU_SE_percent=np.where(mask,rel_se['GPU'],np.nan).astype('f4'),difference_SE_pp=np.where(mask,diff_se,np.nan).astype('f4'),
    evaluated=mask,x_mm=x,depth_mm=z)

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,2,figsize=(13,10),constrained_layout=True)
xs=abs(x)<=25;zs=z<=140;extent=[x[xs][0]-.05,x[xs][-1]+.05,z[zs][-1]+.125,0]
for ax,engine in zip(axs[0],['TOPAS','GPU']):
    im=ax.imshow(np.where(mask,rel_se[engine],np.nan)[zs][:,xs],extent=extent,aspect='auto',vmin=0,vmax=8,cmap='magma')
    ax.set(title=engine+' dose: estimated relative standard error',xlabel='Transverse x (mm)',ylabel='Water depth (mm)')
    fig.colorbar(im,ax=ax,label='1-sigma SE (%)',extend='max')
ax=axs[1,0]
for key,color,label in [('inner_field_peaks','#2563eb','7 inner peaks'),('inner_field_valleys','#e87924','6 inner valleys')]:
    for engine,style in [('TOPAS','-'),('GPU','--')]:
        ax.plot(z,layers[key][engine+'_SE_percent'],color=color,ls=style,lw=.8,label=label+' / '+engine)
ax.set(xlim=(0,125),ylim=(0,3),xlabel='Water depth (mm)',ylabel='ROI mean dose SE (%)',title='Each 0.25 mm layer: average ROI inside each batch')
ax.legend(fontsize=8);ax.grid(alpha=.2)
ax=axs[1,1]
for key,label,color in [('voxel_difference_noise_RMS_pp','Estimated noise in GPU - TOPAS','#2563eb'),('observed_voxel_difference_RMS_percent','Observed GPU - TOPAS','#e87924')]:
    ax.plot(z,[r[key] for r in depth_rows],label=label,color=color,lw=.9)
ax.set(xlim=(0,125),ylim=(0,10),xlabel='Water depth (mm)',ylabel='Relative voxel difference RMS (%)',title='No ROI averaging; same 3% reference cutoff')
ax.legend(fontsize=8);ax.grid(alpha=.2)
fig.suptitle('Monte Carlo uncertainty | 12.8M histories per engine (4 independent 3.2M batches)\nSE of the final four-batch average; scoring grid 0.1 x 100 x 0.25 mm')
fig.savefig(D/'mc_noise.png',dpi=170);plt.close(fig)

report=['# 当前 1280 万粒子的蒙卡统计误差','',
    'TOPAS 与 GPU 各 4 批、每批 320 万源粒子，随机种子互异。以下 1σ 均为最终四批平均剂量的标准误差 SE = std(四批剂量, ddof=1)/sqrt(4)，不是单批波动。两引擎独立，剂量差的方差相加。','',
    '## 逐体素统计误差','',
    '峰谷数字取 20≤深度<120 mm，使用原有横向 ROI，并叠加 TOPAS≥3% Dmax 掩膜；表中为体素 SE 的中位数，未经峰谷空间平均。','',
    '| 区域 | TOPAS 1σ | GPU 1σ | 剂量差合成 1σ | 剂量差 1σ 的 P90 |','|---|---:|---:|---:|---:|']
for key,label in [('all_gamma_voxels','全部 gamma 体素'),('inner_field_peaks_20_120mm_voxels','内部峰区'),('inner_field_valleys_20_120mm_voxels','内部谷区')]:
    st=groups[key];report.append(f'| {label} | {st["TOPAS_SE_percent"]["p50"]:.3f}% | {st["GPU_SE_percent"]["p50"]:.3f}% | {st["difference_SE_pp"]["p50"]:.3f}% | {st["difference_SE_pp"]["p90"]:.3f}% |')
st=groups['all_gamma_voxels']
report+=['',f'全部纳入体素的实际相对剂量差 RMS 为 {st["observed_difference_percent"]["rms"]:.3f}%，批间方差预测的统计噪声 RMS 为 {st["difference_SE_pp"]["rms"]:.3f}%。这用于判断波动量级，不是对系统偏差的唯一分解。','',
    '## 每层峰谷 ROI 平均值的误差','',
    '每批先在 ROI 内平均，再计算批间方差，保留同一条粒子轨迹引起的空间相关。表中为 20–120 mm 各 0.25 mm 深度层的中位数。','',
    '| ROI | TOPAS 1σ | GPU 1σ | GPU/TOPAS−1 的 1σ | 当前近似 95% CI 半宽 |','|---|---:|---:|---:|---:|']
for key,label in [('central_peak','中央峰'),('adjacent_valleys','邻近谷'),('inner_field_peaks','内部 7 峰合并'),('inner_field_valleys','内部 6 谷合并')]:
    st=layer_summaries[key];report.append(f'| {label} | {st["TOPAS_SE_percent"]["p50"]:.3f}% | {st["GPU_SE_percent"]["p50"]:.3f}% | {st["ratio_SE_pp"]["p50"]:.3f}% | ±{st["CI95_half_width_pp"]["p50"]:.3f}% |')
report+=['','## 同引擎、不同随机数的交叉检查','',
    '将每个引擎四批分为两组，两组各 640 万粒子，直接计算同引擎 gamma。三种分组彼此重叠，不视作独立重复实验。每组自己的参考剂量≥3% Dmax；0.2 mm 搜索步长 0.02 mm。该检查没有 TOPAS/GPU 模型差异，但每侧粒子数仅为正式比较的一半，统计差异约放大 sqrt(2)，通过率不可直接线性换算。','',
    '| 引擎 | Global 3%/0 mm | Local 3%/0 mm | Global 3%/0.2 mm | Local 3%/0.2 mm |','|---|---:|---:|---:|---:|']
for engine in data:
    rr=[v for v in self_checks if v['engine']==engine]
    cells=[f'{min(v[k] for v in rr):.2f}–{max(v[k] for v in rr):.2f}%' for k in ['global_0mm_pass_percent','local_0mm_pass_percent','global_0p2mm_pass_percent','local_0p2mm_pass_percent']]
    report.append('| '+engine+' | '+' | '.join(cells)+' |')
report+=['','## 若希望统计误差达到 1%','',
    '以下按 N_new=N_current×(SE_current/SE_target)^2 粗估，每个引擎同时增加粒子数、网格与模型不变。95% 目标使用未来较多独立批次的正态近似 1.96；不是四批现有 t 区间的保证，也不能消除模型偏差。','',
    '| 目标量 | 当前典型 1σ | 达到 1σ≤1%，每引擎总粒子数 | 达到约 95% 半宽≤1%，每引擎总粒子数 |','|---|---:|---:|---:|']
for key,label in [('inner_field_peaks_20_120mm_voxels','峰区典型单体素差值'),('inner_field_valleys_20_120mm_voxels','谷区典型单体素差值')]:
    st=planning[key]['p50'];report.append(f'| {label} | {st["current_difference_SE_pp"]:.3f}% | {st["histories_per_engine_for_1pp_SE"]/1e6:.1f} M | {st["histories_per_engine_for_1pp_95_normal_halfwidth"]/1e6:.1f} M |')
st=planning['inner_field_valleys_layer_ROI_median']
report.append(f'| 每层 6 谷平均值之比 | {st["current_ratio_SE_pp"]:.3f}% | {st["histories_per_engine_for_1pp_SE"]/1e6:.1f} M | {st["histories_per_engine_for_1pp_95_normal_halfwidth"]/1e6:.1f} M |')
report+=['',
    '只有 4 批，每个引擎单体素方差只有 3 个自由度，以上估计存在较大不确定性；不将许多相关体素当作独立样本来缩窄置信区间。1σ≤1% 与 95% 误差范围≤±1% 是不同目标。',
    'gamma 空间搜索可能容纳统计波动或位置差异，不能用较高通过率证明逐体素误差已小于 1%。入口处已有显著峰区差异，仍需与统计噪声分开处理。',
    '当前入口第一层（0–0.25 mm）按固定 ROI 的近似 Welch-t 95% 区间：','']
for key,label in [('inner_field_peaks','内部 7 峰'),('inner_field_valleys','内部 6 谷')]:
    row=next(v for v in selected if v['ROI']==key and v['requested_depth_mm']==0)
    report.append(f'- {label}：GPU/TOPAS−1 = {row["difference_percent"]:+.3f}%，95% CI [{row["CI95_low_percent"]:+.3f}%, {row["CI95_high_percent"]:+.3f}%]。')
report+=['','已核对全部 8 份原始剂量 SHA256 和种子，计算直接串行执行，内存上限 30 GB；本次没有追加粒子模拟。']
(D/'MC_NOISE_zh.md').write_text('\n'.join(report)+'\n')
summary['memory_peak_bytes']=int((cg/'memory.peak').read_text())
(D/'noise_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print('COMPLETE',str(D),flush=True)
