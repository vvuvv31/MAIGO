from pathlib import Path
import csv,datetime,hashlib,json,time
import numpy as np
import scipy
import pymedphys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=Path('/mnt/sdb/wuwei/MAIGO_review_fix_20260923')
O=R/'evidence/field3cm_emonly_20260923'
D=O/'results/gamma_0p2mm_3pct_local_global_cutoff3';D.mkdir(exist_ok=True)
OLD=O/'results/gamma_0mm_3pct_local_global_cutoff3'

def spatial_gamma(axes,ref,eva,local):
    return pymedphys.gamma(
        axes,ref,axes,eva,dose_percent_threshold=3.,distance_mm_threshold=.2,
        lower_percent_dose_cutoff=3.,interp_fraction=10,max_gamma=2.,
        local_gamma=local,global_normalisation=float(ref.max()),skip_once_passed=False,
        random_subset=None,ram_available=1024**3,interp_algo='scipy')

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
assert n==201625
previous=json.loads((OLD/'gamma_summary.json').read_text())
old_maps=np.load(OLD/'gamma_maps.npz')
assert np.array_equal(old_maps['evaluated'],mask)
np.testing.assert_array_equal(old_maps['x_mm'],x)
np.testing.assert_array_equal(old_maps['depth_mm'],z)
gammas={};results={}
spatial_rois={'central_peak':abs(x)<.25,'adjacent_valleys':abs(abs(x)-1.8)<.45,
    'inner_field_peaks':np.any(abs(x[None,:]-np.arange(-3,4)[:,None]*3.6)<.25,axis=0),
    'inner_field_valleys':np.any(abs(x[None,:]-(np.arange(-3,3)+.5)[:,None]*3.6)<.45,axis=0)}
for name,local in [('global',False),('local',True)]:
    print(f'Starting {name}: 3% / 0.2 mm, interp_fraction=10 (0.02 mm), {n} voxels',flush=True)
    start=time.monotonic();g=spatial_gamma((z,x),ref,eva,local);wall=time.monotonic()-start
    assert np.array_equal(np.isfinite(g),mask)
    old=old_maps['gamma_'+name]
    assert np.all(g[mask]<=old[mask]+1e-10)
    passing=mask&(g<=1);yes=int(passing.sum())
    old_passing=mask&(old<=1)
    assert not np.any(old_passing&~passing)
    recovered=int((passing&~old_passing).sum())
    old_yes=int(old_passing.sum());old_fail=n-old_yes
    assert old_yes==previous['results'][name]['passing_voxels']
    roi_results={}
    for key,m in spatial_rois.items():
        included=mask&m[None,:];nn=int(included.sum());ny=int((passing&included).sum())
        roi_results[key]=dict(evaluated_voxels=nn,passing_voxels=ny,failing_voxels=nn-ny,pass_rate_percent=100*ny/nn if nn else None)
    results[name]=dict(local_gamma=local,evaluated_voxels=n,passing_voxels=yes,failing_voxels=n-yes,
        pass_rate_percent=100*yes/n,calculation_wall_s=wall,
        gamma_statistics=dict(mean=float(g[mask].mean()),median=float(np.median(g[mask])),p95=float(np.percentile(g[mask],95)),p99=float(np.percentile(g[mask],99)),maximum=float(g[mask].max())),
        zero_distance_comparison=dict(pass_rate_percent=100*old_yes/n,passing_voxels=old_yes,
            newly_passing_voxels=recovered,pass_rate_increase_percentage_points=100*recovered/n,
            previously_failing_now_passing_percent=100*recovered/old_fail,
            previously_passing_now_failing_voxels=0,all_gamma_values_no_larger_than_zero_distance=True),
        denominator='0.03 * TOPAS dose at the same voxel' if local else '0.03 * TOPAS global maximum dose',rois=roi_results)
    gammas[name]=g
    print(f'{name}: {100*yes/n:.6f}% pass ({yes}/{n}), {wall:.2f} seconds',flush=True)

assert np.all(gammas['local'][mask]+1e-12>=gammas['global'][mask])
assert np.all((gammas['local'][mask]>1)|(gammas['global'][mask]<=1))
cg=next(s.split('::',1)[1] for s in Path('/proc/self/cgroup').read_text().splitlines() if s.startswith('0::'))
cg=Path('/sys/fs/cgroup')/cg.lstrip('/')
assert int((cg/'memory.max').read_text())<=30000000000 and (cg/'memory.swap.max').read_text().strip()=='0'
summary=dict(status='COMPLETE',completed_at=datetime.datetime.now().astimezone().isoformat(),
    pymedphys_version=pymedphys.__version__,numpy_version=np.__version__,scipy_version=scipy.__version__,
    reference='TOPAS',evaluation='GPU',histories_per_engine=M['histories_per_engine'],
    distance_criterion_mm=.2,dose_difference_criterion_percent=3.,lower_reference_dose_cutoff_percent=3.,
    search_step_mm=.02,interp_fraction=10,
    cutoff_definition='Both modes use the identical reference mask: TOPAS >= 0.03 * TOPAS.max()',
    reference_max_dose_Gy_per_source=dmax,reference_dose_cutoff_Gy_per_source=cutoff,
    global_absolute_dose_tolerance_Gy_per_source=.03*dmax,
    local_absolute_tolerance_range_Gy_per_source=[float(.03*ref[mask].min()),float(.03*ref[mask].max())],
    total_grid_voxels=int(mask.size),evaluated_voxels=n,excluded_voxels=int(mask.size-n),results=results,
    implementation=dict(method='Public pymedphys.gamma; 2D Euclidean distance in transverse x and water depth.',
        maximum_search_radius_mm=.4,max_gamma=2.,
        search_step_note='distance_mm_threshold / interp_fraction = 0.2 / 10 = 0.02 mm; maximum radius 0.4 mm keeps the adaptive radial step at 0.02 mm. Angular spacing is selected by PyMedPhys, not a Cartesian search grid.',
        interpolation='scipy multilinear interpolation',all_voxels_checked=True,gamma_output_clipped=True,
        clipping_note='Gamma values above 2 are capped at 2; this does not change classification at gamma <= 1. Reported moments are for capped values.',
        local_passing_is_subset_of_global=True,all_previous_passing_voxels_still_pass=True,
        input_sha256_matches_completed_mc_run=True),
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
    fig.colorbar(im,ax=ax,label='Gamma (values capped at 2)')
ax=fig.add_subplot(gs[1,:])
for name,color in [('global','#2563eb'),('local','#ea7b24')]:
    yy=np.array([r[name+'_pass_rate_percent'] if r[name+'_pass_rate_percent'] is not None else np.nan for r in depth_rows])
    ax.plot(z,yy,label=name.capitalize()+' (0.2 mm)',color=color,lw=1.2)
    old_pass=np.sum(mask&(old_maps['gamma_'+name]<=1),axis=1)
    counts=mask.sum(axis=1)
    yy_old=np.divide(100.*old_pass,counts,out=np.full(z.shape,np.nan),where=counts>0)
    ax.plot(z,yy_old,label=name.capitalize()+' (0 mm)',color=color,lw=.7,alpha=.5,ls='--')
ax.set(xlim=(0,140),ylim=(0,100.5),xlabel='Water depth (mm)',ylabel='Passing voxels (%)',title='Pass rate in each depth layer; same reference-dose mask')
ax.legend();ax.grid(alpha=.2)
fig.suptitle('PyMedPhys | 3% / 0.2 mm | Search step: 0.02 mm | TOPAS cutoff: 3% of Dmax\n12.8M histories per engine; 2D x-depth gamma; 100 mm vertical averaging')
fig.savefig(D/'gamma_local_global.png',dpi=170);plt.close(fig)
report=['# Gamma：3% / 0.2 mm，搜索步长 0.02 mm','',
    '两种模式均以 TOPAS 为参考、GPU 为评价，低剂量阈值均为 TOPAS 最大剂量的 3%，纳入完全相同的体素。',
    '','| 模式 | 3% / 0 mm 通过率 | 3% / 0.2 mm 通过率 | 提升（百分点） | 未通过体素 |','|---|---:|---:|---:|---:|']
for name,r in results.items():
    old=r['zero_distance_comparison']
    report.append(f'| {name.capitalize()} | {old["pass_rate_percent"]:.6f}% | {r["pass_rate_percent"]:.6f}% | {old["pass_rate_increase_percentage_points"]:.6f} | {r["failing_voxels"]:,} |')
report+=['',f'共同评估体素：{n:,}；两端各 1280 万源历史，采用四批平均剂量；γ ≤ 1 通过，没有抽样或拟合归一化。','',
    '- γ 是候选点上 sqrt((剂量差/剂量容限)^2 + (空间距离/0.2 mm)^2) 的最小值。',
    '- Global 剂量容限：0.03×TOPAS全局最大剂量。',
    '- Local 剂量容限：0.03×参考体素TOPAS剂量，搜索时固定参考点的容限。',
    '','Local 在谷区采用更小的绝对剂量差容限，因此能更严格地反映谷区的相对误差。这里保留了蒙卡统计波动，未对通过率估计置信区间。','',
    '谷区若只有全局峰值的 5%，Local 容限仅相当于全局峰值的 0.15%，是 Global 容限的 1/20。全量程剖面图很难显示这么小的差异；峰谷 ROI 平均也会弱化逐体素的统计波动。',
    '空间搜索可容纳细小位移，也可能在陡梯度或噪声场中找到较接近的剂量。通过率提升本身不能区分系统性位移、统计噪声与物理模型误差。','',
    f'本次使用 PyMedPhys {pymedphys.__version__} 公开 gamma 接口，distance_mm_threshold=0.2、interp_fraction=10，因此径向搜索步长为 0.02 mm；使用 scipy 多线性插值。搜索为 x–深度二维欧氏距离，原始网格 0.1×0.25 mm，垂直方向平均 100 mm，不是完整三维 gamma。0.02 mm 是插值搜索步长，并非原始模拟的空间分辨率。',
    'max_gamma=2，最大搜索半径 0.4 mm；大于 2 的 gamma 保存为 2，不影响 γ≤1 的通过判定。角向采样由 PyMedPhys 决定。所有体素均计算，无随机抽样。',
    '已核对 8 份输入剂量的 SHA256、同一阈值掩膜和坐标；逐点 gamma 不大于对应零距离值，所有此前通过的体素仍然通过，Local 通过集合为 Global 的子集。','',
    '## 峰谷 ROI 补充结果','',
    '下表使用与前述 field 分析相同的横向 ROI，并叠加同一 3% 参考剂量阈值，涵盖全部满足阈值的深度层。','',
    '| ROI | 评估体素 | Global 通过率 | Local 通过率 |','|---|---:|---:|---:|']
for key,label in [('central_peak','中央峰'),('adjacent_valleys','邻近谷'),('inner_field_peaks','内部 7 峰合并'),('inner_field_valleys','内部 6 谷合并')]:
    a=results['global']['rois'][key];b=results['local']['rois'][key]
    report.append(f'| {label} | {a["evaluated_voxels"]:,} | {a["pass_rate_percent"]:.4f}% | {b["pass_rate_percent"]:.4f}% |')
(D/'GAMMA_RESULT_zh.md').write_text('\n'.join(report)+'\n')
print(json.dumps(summary,indent=2),flush=True)
