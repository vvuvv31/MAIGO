from pathlib import Path
from dataclasses import replace
import csv,datetime,hashlib,json,os
import numpy as np
import pymedphys
from pymedphys._gamma.implementation import shell
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/mnt/sdb/wuwei/MAIGO_review_fix_20260923')
OLD=R/'evidence/field3cm_emonly_20260923'
O=Path(os.environ.get('MAIGO_ANALYSIS_INPUT',str(R/'evidence/field3cm_emonly_64M_20260923')))
D=Path(os.environ.get('MAIGO_ANALYSIS_OUTPUT',str(O/'results')))
M=json.loads((O/'manifest.json').read_text());U=json.loads((D/'uncertainty_details.json').read_text())
B=M['batches'];N=M['histories_per_engine'];data={'TOPAS':[],'GPU':[]}
for v in U['dose_files']:
    raw=Path(v['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==v['sha256']
    data[v['engine']].append(np.frombuffer(raw,dtype='<f8' if v['engine']=='TOPAS' else '<f4').reshape(1000,1000).astype(float)/M['histories_per_batch'])
data={k:np.stack(v) for k,v in data.items()};assert all(len(v)==B for v in data.values())
mean={k:v.mean(0) for k,v in data.items()};vm={k:v.var(0,ddof=1)/B for k,v in data.items()}
ref=mean['TOPAS'];eva=mean['GPU'];dmax=float(ref.max());mask=ref>=.03*dmax
x=-49.95+np.arange(1000)*.1;z=.125+np.arange(1000)*.25;axes=(z,x)
gammas={};results={}
for mode,local in [('global',False),('local',True)]:
    opt=shell.GammaInternalFixedOptions.from_user_inputs(axes,ref,axes,eva,3.,1.,
        lower_percent_dose_cutoff=3.,interp_fraction=10,max_gamma=2.,local_gamma=local,
        global_normalisation=dmax,skip_once_passed=False,random_subset=None,ram_available=1024**3,interp_algo='scipy')
    zero=shell.gamma_loop(replace(opt,maximum_test_distance=0.))[:,0,0].reshape(ref.shape)
    zero[np.isinf(zero)]=np.nan
    denominator=.03*ref[mask] if local else .03*dmax
    np.testing.assert_allclose(zero[mask],abs(eva[mask]-ref[mask])/denominator,rtol=1e-12,atol=1e-12)
    spatial=pymedphys.gamma(axes,ref,axes,eva,3.,.2,lower_percent_dose_cutoff=3.,interp_fraction=10,
        max_gamma=2.,local_gamma=local,global_normalisation=dmax,skip_once_passed=False,
        random_subset=None,ram_available=1024**3,interp_algo='scipy')
    assert np.array_equal(np.isfinite(zero),mask) and np.array_equal(np.isfinite(spatial),mask)
    assert np.all(spatial[mask]<=zero[mask]+1e-10)
    for distance,g in [('0mm',zero),('0p2mm',spatial)]:
        key=mode+'_'+distance;gammas[key]=g;n=int(mask.sum());yes=int((mask&(g<=1)).sum())
        results[key]=dict(evaluated_voxels=n,passing_voxels=yes,failing_voxels=n-yes,pass_rate_percent=100*yes/n)
assert np.all(gammas['local_0p2mm'][mask]+1e-10>=gammas['global_0p2mm'][mask])
def relative(a,b):return np.divide(a,b,out=np.full_like(a,np.nan),where=b>0)
se={k:100*relative(np.sqrt(vm[k]),mean[k]) for k in data}
diff_se=100*relative(np.sqrt(vm['TOPAS']+vm['GPU']),ref)
delta=100*relative(eva-ref,ref)
peak=np.any(abs(x[None,:]-np.arange(-3,4)[:,None]*3.6)<.25,axis=0)
valley=np.any(abs(x[None,:]-(np.arange(-3,3)+.5)[:,None]*3.6)<.45,axis=0)
mid=(z>=20)&(z<120)
def stats(v):return dict(median=float(np.median(v)),p90=float(np.percentile(v,90)),rms=float(np.sqrt(np.mean(v*v))))
noise={}
for key,m in [('all_gamma_voxels',mask),('inner_peak_voxels_20_120mm',mask&mid[:,None]&peak[None,:]),('inner_valley_voxels_20_120mm',mask&mid[:,None]&valley[None,:])]:
    noise[key]=dict(voxels=int(m.sum()),TOPAS_SE_percent=stats(se['TOPAS'][m]),GPU_SE_percent=stats(se['GPU'][m]),
        difference_SE_pp=stats(diff_se[m]),observed_difference_percent=stats(delta[m]))
old0=json.loads((OLD/'results/gamma_0mm_3pct_local_global_cutoff3/gamma_summary.json').read_text())
old2=json.loads((OLD/'results/gamma_0p2mm_3pct_local_global_cutoff3/gamma_summary.json').read_text())
baseline={}
for mode in ['global','local']:
    baseline[mode+'_0mm']=old0['results'][mode]['pass_rate_percent']
    baseline[mode+'_0p2mm']=old2['results'][mode]['pass_rate_percent']
summary=dict(status='COMPLETE',histories_per_engine=N,batches_per_engine=B,reference='TOPAS',evaluation='GPU',
    dose_criterion_percent=3.,distance_criteria_mm=[0.,.2],search_step_mm=.02,interp_fraction=10,
    lower_reference_dose_cutoff_percent=3.,reference_Dmax_Gy_per_source=dmax,results=results,
    noise=noise,baseline_12p8M_gamma_pass_rates=baseline,
    baseline_comparison_note='Each particle count uses its own TOPAS mean >=3% Dmax cutoff; the evaluated voxel count may change. Old 4 batches are part of the 20-batch result, so the two estimates are correlated.',
    gamma_method='0 mm: exact zero-radius PyMedPhys core, verified analytically. 0.2 mm: public pymedphys.gamma, 2D x-depth, scipy linear interpolation, max_gamma=2; >2 values capped without affecting pass classification.',
    noise_method='Unbiased variance across independent batches divided by batch count estimates variance of the final mean. Difference variance sums independent engine variances; percentages use TOPAS mean as reporting scale.',
    source_histories_normalisation=True,no_fitted_normalisation=True,all_voxels_evaluated=True,
    gamma_version=pymedphys.__version__)
(D/'gamma_noise_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
np.savez_compressed(D/'gamma_maps.npz',**gammas,evaluated=mask,x_mm=x,depth_mm=z)
np.savez_compressed(D/'noise_maps.npz',TOPAS_SE_percent=np.where(mask,se['TOPAS'],np.nan).astype('f4'),
    GPU_SE_percent=np.where(mask,se['GPU'],np.nan).astype('f4'),difference_SE_pp=np.where(mask,diff_se,np.nan).astype('f4'),evaluated=mask,x_mm=x,depth_mm=z)
depth_rows=[]
for k,dep in enumerate(z):
    n=int(mask[k].sum());row=dict(depth_mm=float(dep),evaluated_voxels=n)
    for key,g in gammas.items():row[key+'_pass_percent']=float(np.mean(g[k,mask[k]]<=1)*100) if n else None
    row['expected_noise_RMS_pp']=float(np.sqrt(np.mean(diff_se[k,mask[k]]**2))) if n else None
    row['observed_difference_RMS_percent']=float(np.sqrt(np.mean(delta[k,mask[k]]**2))) if n else None
    depth_rows.append(row)
with (D/'gamma_noise_by_depth.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(depth_rows[0]));w.writeheader();w.writerows(depth_rows)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,2,figsize=(13,10),constrained_layout=True)
xs=abs(x)<=25;zs=z<=140;extent=[x[xs][0]-.05,x[xs][-1]+.05,z[zs][-1]+.125,0]
for ax,mode in zip(axs[0],['global','local']):
    key=mode+'_0p2mm';im=ax.imshow(gammas[key][zs][:,xs],extent=extent,aspect='auto',vmin=0,vmax=2,cmap='viridis')
    ax.set(title=f'{mode.capitalize()} 3% / 0.2 mm: {results[key]["pass_rate_percent"]:.4f}% pass',xlabel='Transverse x (mm)',ylabel='Water depth (mm)')
    fig.colorbar(im,ax=ax,label='Gamma (capped at 2)')
ax=axs[1,0]
for mode,color in [('global','#2563eb'),('local','#ea7b24')]:
    for distance,style in [('0mm','--'),('0p2mm','-')]:
        ax.plot(z,[v[mode+'_'+distance+'_pass_percent'] for v in depth_rows],color=color,ls=style,label=mode+' / '+distance,lw=.9)
ax.set(xlim=(0,135),ylim=(0,100.5),xlabel='Water depth (mm)',ylabel='Gamma pass rate (%)');ax.legend(fontsize=8);ax.grid(alpha=.2)
ax=axs[1,1]
for key,label,color in [('expected_noise_RMS_pp','Estimated MC noise','#2563eb'),('observed_difference_RMS_percent','Observed GPU - TOPAS','#ea7b24')]:
    ax.plot(z,[v[key] for v in depth_rows],label=label,color=color,lw=.9)
ax.set(xlim=(0,125),xlabel='Water depth (mm)',ylabel='Relative voxel difference RMS (%)');ax.legend(fontsize=8);ax.grid(alpha=.2)
fig.suptitle(f'{N/1e6:g}M histories per engine | {B} independent batches | TOPAS cutoff: 3% Dmax\n2D gamma; 0.02 mm search step; scoring grid 0.1 x 100 x 0.25 mm')
fig.savefig(D/'gamma_noise.png',dpi=170);plt.close(fig)
report=[f'# {N/1e6:g}M 粒子/引擎：Gamma 与统计误差','',
    f'每引擎 {B} 个独立批次，每批 320 万；TOPAS 为参考，低剂量阈值 3% Dmax，共 {int(mask.sum()):,} 个评估体素。3%/0.2 mm 的搜索步长为 0.02 mm，二维横向—深度搜索。','',
    '| 模式 | 原 12.8M 通过率 | 当前通过率 | 未通过体素 |','|---|---:|---:|---:|']
for key,r in results.items():report.append(f'| {key} | {baseline[key]:.4f}% | {r["pass_rate_percent"]:.4f}% | {r["failing_voxels"]:,} |')
report+=['','两次粒子数采用各自 TOPAS 均值的 3% Dmax 掩膜，体素数可不同；新结果包含原四批，因此新旧估计相关。','',
    '| 区域 | TOPAS 1σ 中位数 | GPU 1σ 中位数 | 差值合成 1σ 中位数 |','|---|---:|---:|---:|']
for key,label in [('all_gamma_voxels','全部 gamma 体素'),('inner_peak_voxels_20_120mm','20–120 mm 内部峰区'),('inner_valley_voxels_20_120mm','20–120 mm 内部谷区')]:
    r=noise[key];report.append(f'| {label} | {r["TOPAS_SE_percent"]["median"]:.3f}% | {r["GPU_SE_percent"]["median"]:.3f}% | {r["difference_SE_pp"]["median"]:.3f}% |')
report+=['','以上为最终平均剂量的逐体素标准误差，不是单批波动或 95% 误差界限；峰谷 ROI 的平均剂量、置信区间和随深度曲线另见 RESULTS.md、uncertainty_details.json。',
    '增加粒子数降低统计误差，但不消除系统性的模型差异。没有拟合归一化或改变物理参数。']
(D/'GAMMA_NOISE_zh.md').write_text('\n'.join(report)+'\n')
print(json.dumps(dict(histories_per_engine=N,gamma=results,noise=noise),indent=2),flush=True)
