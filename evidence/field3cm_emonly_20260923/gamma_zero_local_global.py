from pathlib import Path
from dataclasses import replace
import csv,datetime,hashlib,json,time
import numpy as np
import scipy
import pymedphys
from pymedphys._gamma.implementation import shell
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=Path('/mnt/sdb/wuwei/MAIGO_review_fix_20260923')
O=R/'evidence/field3cm_emonly_20260923'
D=O/'results/gamma_0mm_3pct_local_global_cutoff3';D.mkdir(exist_ok=True)

def zero_gamma(axes,ref,eva,local):
    options=shell.GammaInternalFixedOptions.from_user_inputs(
        axes,ref,axes,eva,dose_percent_threshold=3.,distance_mm_threshold=1.,
        lower_percent_dose_cutoff=3.,interp_fraction=10,max_gamma=2.,
        local_gamma=local,global_normalisation=float(ref.max()),skip_once_passed=False,
        random_subset=None,ram_available=1024**3,interp_algo='scipy')
    # Search only the coincident coordinate. Its spatial contribution is 0;
    # the positive internal spatial denominator does not allow displacement.
    options=replace(options,maximum_test_distance=0.)
    gamma=shell.gamma_loop(options)[:,0,0].reshape(ref.shape)
    gamma[np.isinf(gamma)]=np.nan
    return gamma

test_axes=(np.array([-1.,1.]),np.array([-1.,1.]))
test_ref=np.array([[1.,.5],[.02,.8]])
test_eval=np.array([[1.,.52],[.8,.84]])
for local,expected_values in [(False,[0.,2/3,4/3]),(True,[0.,4/3,5/3])]:
    test=zero_gamma(test_axes,test_ref,test_eval,local)
    assert np.isnan(test[1,0])
    np.testing.assert_allclose(test[[0,0,1],[0,1,1]],expected_values,atol=1e-12)

M=json.loads((O/'manifest.json').read_text())
U=json.loads((O/'results/uncertainty_details.json').read_text())
assert json.loads((O/'serial_status.json').read_text())['status']=='COMPLETE'
expected={v['path']:v['sha256'] for v in U['dose_files']}
ref=np.zeros((1000,1000));eva=np.zeros_like(ref);inputs=[]
for i in range(1,5):
    td=Path(M['cases'][f'topas_b{i}']);gd=Path(M['cases'][f'gpu_b{i}']).parent
    tq=json.loads((td/'quality.json').read_text());gq=json.loads((gd/'quality.json').read_text())
    assert tq['accepted'] and gq['accepted'] and tq['histories']==gq['histories']==M['histories_per_batch']
    for engine,p,dt,dest in [('TOPAS',td/'dose.bin','<f8',ref),('GPU',Path(gq['dose_path']),'<f4',eva)]:
        raw=p.read_bytes();h=hashlib.sha256(raw).hexdigest();assert h==expected[str(p)]
        dest+=np.frombuffer(raw,dtype=dt).reshape(1000,1000).astype(float)/M['histories_per_engine']
        inputs.append(dict(engine=engine,batch=i,path=str(p),sha256=h))

z=.125+np.arange(1000)*.25;x=-49.95+np.arange(1000)*.1
dmax=float(ref.max());cutoff=.03*dmax;mask=ref>=cutoff;n=int(mask.sum())
assert n>0
gammas={};results={}
spatial_rois={'central_peak':abs(x)<.25,'adjacent_valleys':abs(abs(x)-1.8)<.45,
    'inner_field_peaks':np.any(abs(x[None,:]-np.arange(-3,4)[:,None]*3.6)<.25,axis=0),
    'inner_field_valleys':np.any(abs(x[None,:]-(np.arange(-3,3)+.5)[:,None]*3.6)<.45,axis=0)}
for name,local in [('global',False),('local',True)]:
    start=time.monotonic();g=zero_gamma((z,x),ref,eva,local);wall=time.monotonic()-start
    assert np.array_equal(np.isfinite(g),mask)
    denominator=.03*ref[mask] if local else .03*dmax
    exact=np.abs(eva[mask]-ref[mask])/denominator
    np.testing.assert_allclose(g[mask],exact,rtol=1e-12,atol=1e-12)
    assert np.array_equal(g[mask]<=1,exact<=1)
    passing=mask&(g<=1);yes=int(passing.sum())
    roi_results={}
    for key,m in spatial_rois.items():
        included=mask&m[None,:];nn=int(included.sum());ny=int((passing&included).sum())
        roi_results[key]=dict(evaluated_voxels=nn,passing_voxels=ny,failing_voxels=nn-ny,pass_rate_percent=100*ny/nn if nn else None)
    results[name]=dict(local_gamma=local,evaluated_voxels=n,passing_voxels=yes,failing_voxels=n-yes,
        pass_rate_percent=100*yes/n,calculation_wall_s=wall,
        gamma_statistics=dict(mean=float(g[mask].mean()),median=float(np.median(g[mask])),p95=float(np.percentile(g[mask],95)),p99=float(np.percentile(g[mask],99)),maximum=float(g[mask].max())),
        max_abs_difference_vs_analytic=float(np.max(abs(g[mask]-exact))),classification_mismatches_vs_analytic=0,
        denominator='0.03 * TOPAS dose at the same voxel' if local else '0.03 * TOPAS global maximum dose',rois=roi_results)
    gammas[name]=g

assert np.all(gammas['local'][mask]+1e-12>=gammas['global'][mask])
assert np.all((gammas['local'][mask]>1)|(gammas['global'][mask]<=1))
previous=json.loads((O/'results/gamma_0mm_3pct_cutoff3/gamma_summary.json').read_text())
for key in ['evaluated_voxels','passing_voxels','failing_voxels','pass_rate_percent']:
    assert results['global'][key]==previous[key],key
cg=next(s.split('::',1)[1] for s in Path('/proc/self/cgroup').read_text().splitlines() if s.startswith('0::'))
cg=Path('/sys/fs/cgroup')/cg.lstrip('/')
assert int((cg/'memory.max').read_text())<=30000000000 and (cg/'memory.swap.max').read_text().strip()=='0'
summary=dict(status='COMPLETE',completed_at=datetime.datetime.now().astimezone().isoformat(),
    pymedphys_version=pymedphys.__version__,numpy_version=np.__version__,scipy_version=scipy.__version__,
    reference='TOPAS',evaluation='GPU',histories_per_engine=M['histories_per_engine'],
    distance_criterion_mm=0.,dose_difference_criterion_percent=3.,lower_reference_dose_cutoff_percent=3.,
    cutoff_definition='Both modes use the identical reference mask: TOPAS >= 0.03 * TOPAS.max()',
    reference_max_dose_Gy_per_source=dmax,reference_dose_cutoff_Gy_per_source=cutoff,
    global_absolute_dose_tolerance_Gy_per_source=.03*dmax,
    local_absolute_tolerance_range_Gy_per_source=[float(.03*ref[mask].min()),float(.03*ref[mask].max())],
    total_grid_voxels=int(mask.size),evaluated_voxels=n,excluded_voxels=int(mask.size-n),results=results,
    implementation=dict(method='PyMedPhys GammaInternalFixedOptions + gamma_loop; maximum_test_distance=0; no displacement is searched.',
        maximum_search_radius_mm=0.,internal_spatial_denominator_mm=1.,spatial_term='Identically zero because only distance=0 is evaluated.',
        zero_distance_note='The PyMedPhys 0.41.0 public interface errors at distance_mm_threshold=0. The zero-radius core invocation is the exact same-voxel limiting case.',
        interpolation='scipy at identical grid coordinates',all_voxels_checked=True,gamma_output_clipped=False,
        tiny_grid_verification_passed=True,local_passing_is_subset_of_global=True,global_matches_previous_run=True),
    memory_max_bytes=int((cg/'memory.max').read_text()),memory_peak_bytes=int((cg/'memory.peak').read_text()),
    scoring_scope='Original 0.1 x 100 x 0.25 mm grid, averaged across the 100 mm vertical dimension.',
    monte_carlo_uncertainty_note='Gamma of the four-batch averaged dose. Monte Carlo noise is retained; no pass-rate confidence interval is estimated.',inputs=inputs)
(D/'gamma_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
np.savez_compressed(D/'gamma_maps.npz',gamma_global=gammas['global'],gamma_local=gammas['local'],evaluated=mask,x_mm=x,depth_mm=z)
depth_rows=[]
for k,depth in enumerate(z):
    nn=int(mask[k].sum());row=dict(depth_mm=float(depth),evaluated_voxels=nn)
    for name,g in gammas.items():
        yes=int(np.count_nonzero(mask[k]&(g[k]<=1)))
        row[name+'_passing_voxels']=yes;row[name+'_pass_rate_percent']=100*yes/nn if nn else None
    depth_rows.append(row)
with (D/'gamma_by_depth.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(depth_rows[0]));w.writeheader();w.writerows(depth_rows)

plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig=plt.figure(figsize=(13,10),constrained_layout=True);gs=fig.add_gridspec(2,2,height_ratios=[2,1])
xs=abs(x)<=25;zs=z<=140;extent=[x[xs][0]-.05,x[xs][-1]+.05,z[zs][-1]+.125,0]
for j,name in enumerate(['global','local']):
    ax=fig.add_subplot(gs[0,j]);g=gammas[name]
    im=ax.imshow(g[zs][:,xs],extent=extent,aspect='auto',vmin=0,vmax=2,cmap='viridis')
    ax.set(title=f'{name.capitalize()}: {results[name]["pass_rate_percent"]:.4f}% pass',xlabel='Transverse x (mm)',ylabel='Water depth (mm)')
    fig.colorbar(im,ax=ax,label='Gamma (color scale capped at 2)',extend='max')
ax=fig.add_subplot(gs[1,:])
for name,color in [('global','#2563eb'),('local','#ea7b24')]:
    yy=np.array([r[name+'_pass_rate_percent'] if r[name+'_pass_rate_percent'] is not None else np.nan for r in depth_rows])
    ax.plot(z,yy,label=name.capitalize(),color=color,lw=1)
ax.set(xlim=(0,140),ylim=(0,100.5),xlabel='Water depth (mm)',ylabel='Passing voxels (%)',title='Pass rate in each depth layer; same reference-dose mask')
ax.legend();ax.grid(alpha=.2)
fig.suptitle('PyMedPhys | 0 mm / 3% | Global and local | TOPAS reference cutoff: 3% of Dmax\n12.8M histories per engine; exact zero-radius search; 100 mm vertical averaging')
fig.savefig(D/'gamma_local_global.png',dpi=170);plt.close(fig)
report=['# Gamma：0 mm / 3%，Global 与 Local 对比','',
    '两种模式均以 TOPAS 为参考、GPU 为评价，低剂量阈值均为 TOPAS 最大剂量的 3%，纳入完全相同的体素。',
    '','| 模式 | 通过率 | 通过体素 | 未通过体素 |','|---|---:|---:|---:|']
for name,r in results.items():report.append(f'| {name.capitalize()} | {r["pass_rate_percent"]:.6f}% | {r["passing_voxels"]:,} | {r["failing_voxels"]:,} |')
report+=['',f'共同评估体素：{n:,}；两端各 1280 万源历史，采用四批平均剂量；γ ≤ 1 通过，没有抽样或拟合归一化。','',
    '- Global：γ = |GPU−TOPAS| / (0.03×TOPAS全局最大剂量)。',
    '- Local：γ = |GPU−TOPAS| / (0.03×该体素TOPAS剂量)。',
    '','Local 在谷区采用更小的绝对剂量差容限，因此能更严格地反映谷区的相对误差。这里保留了蒙卡统计波动，未对通过率估计置信区间。','',
    f'PyMedPhys {pymedphys.__version__} 公开接口不能直接处理零距离；调用其 gamma 核心并将搜索半径严格限制为零，两种模式均与同体素解析式逐点核对，分类不一致体素为零。Global 重新计算与上一次结果完全一致。','',
    '## 峰谷 ROI 补充结果','',
    '下表使用与前述 field 分析相同的横向 ROI，并叠加同一 3% 参考剂量阈值，涵盖全部满足阈值的深度层。','',
    '| ROI | 评估体素 | Global 通过率 | Local 通过率 |','|---|---:|---:|---:|']
for key,label in [('central_peak','中央峰'),('adjacent_valleys','邻近谷'),('inner_field_peaks','内部 7 峰合并'),('inner_field_valleys','内部 6 谷合并')]:
    a=results['global']['rois'][key];b=results['local']['rois'][key]
    report.append(f'| {label} | {a["evaluated_voxels"]:,} | {a["pass_rate_percent"]:.4f}% | {b["pass_rate_percent"]:.4f}% |')
(D/'GAMMA_RESULT_zh.md').write_text('\n'.join(report)+'\n')
print(json.dumps(summary,indent=2),flush=True)
