from pathlib import Path
from dataclasses import replace
import csv,datetime,hashlib,inspect,json,time
import numpy as np
import scipy
import pymedphys
from pymedphys._gamma.implementation import shell
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=Path('/mnt/sdb/wuwei/MAIGO_review_fix_20260923')
O=R/'evidence/field3cm_emonly_20260923'
D=O/'results/gamma_0mm_3pct_cutoff3';D.mkdir(exist_ok=True)
DOSE_PERCENT=3.;CUTOFF_PERCENT=3.

def zero_distance_gamma(axes,reference,evaluation,normalisation):
    # The public gamma interface divides by distance_mm_threshold and fails
    # at zero. A zero-radius search is the exact zero-DTA limiting problem.
    # With distance == 0, the positive spatial denominator is immaterial:
    # the spatial term is identically zero and no displaced point is tested.
    options=shell.GammaInternalFixedOptions.from_user_inputs(
        axes,reference,axes,evaluation,dose_percent_threshold=DOSE_PERCENT,
        distance_mm_threshold=1.,lower_percent_dose_cutoff=CUTOFF_PERCENT,
        interp_fraction=10,max_gamma=2.,local_gamma=False,
        global_normalisation=normalisation,skip_once_passed=False,
        random_subset=None,ram_available=1024**3,interp_algo='scipy')
    options=replace(options,maximum_test_distance=0.)
    assert options.maximum_test_distance==0.
    # gamma_loop returns the un-clipped gamma values. max_gamma only sets
    # the unused positive-distance stepping policy for this zero-radius run.
    gamma=shell.gamma_loop(options)[:,0,0].reshape(reference.shape)
    gamma[np.isinf(gamma)]=np.nan
    return gamma

# Verify cutoff, exact matching and nonzero dose differences on a small grid.
test_axes=(np.array([-1.,1.]),np.array([-1.,1.]))
test_ref=np.array([[1.,.5],[.02,.8]])
test_eval=np.array([[1.,.52],[.8,.84]])
test_gamma=zero_distance_gamma(test_axes,test_ref,test_eval,1.)
assert np.isnan(test_gamma[1,0])
np.testing.assert_allclose(test_gamma[[0,0,1],[0,1,1]],[0.,2/3,4/3],atol=1e-12)
same=zero_distance_gamma(test_axes,test_ref,test_ref,1.)
assert np.all(same[np.isfinite(same)]==0)

M=json.loads((O/'manifest.json').read_text())
U=json.loads((O/'results/uncertainty_details.json').read_text())
assert json.loads((O/'serial_status.json').read_text())['status']=='COMPLETE'
assert U['status']=='COMPLETE'
expected={v['path']:v['sha256'] for v in U['dose_files']}
reference=np.zeros((1000,1000));evaluation=np.zeros_like(reference)
inputs=[]
for i in range(1,5):
    td=Path(M['cases'][f'topas_b{i}']);gd=Path(M['cases'][f'gpu_b{i}']).parent
    tq=json.loads((td/'quality.json').read_text());gq=json.loads((gd/'quality.json').read_text())
    assert tq['accepted'] and gq['accepted'] and tq['histories']==gq['histories']==M['histories_per_batch']
    for engine,p,dt,dest in [('TOPAS',td/'dose.bin','<f8',reference),('GPU',Path(gq['dose_path']),'<f4',evaluation)]:
        raw=p.read_bytes();h=hashlib.sha256(raw).hexdigest();assert h==expected[str(p)],p
        dose=np.frombuffer(raw,dtype=dt).reshape(1000,1000)
        dest+=dose.astype(float)/M['histories_per_engine']
        inputs.append(dict(engine=engine,batch=i,path=str(p),sha256=h))
z=.125+np.arange(1000)*.25;x=-49.95+np.arange(1000)*.1
normalisation=float(reference.max());dose_threshold=normalisation*DOSE_PERCENT/100
cutoff=normalisation*CUTOFF_PERCENT/100
mask=reference>=cutoff
start=time.monotonic()
gamma=zero_distance_gamma((z,x),reference,evaluation,normalisation)
elapsed=time.monotonic()-start
assert np.array_equal(np.isfinite(gamma),mask)
exact=np.abs(evaluation-reference)/dose_threshold
max_diff=float(np.max(np.abs(gamma[mask]-exact[mask])))
np.testing.assert_allclose(gamma[mask],exact[mask],rtol=1e-12,atol=1e-12)
assert np.array_equal(gamma[mask]<=1.,exact[mask]<=1.)
passing=mask&(gamma<=1.);failing=mask&~passing
n=int(mask.sum());passed=int(passing.sum());failed=int(failing.sum())
assert n==passed+failed and n>0
norm_z,norm_x=np.unravel_index(np.argmax(reference),reference.shape)
per_depth=[]
for k,depth in enumerate(z):
    count=int(mask[k].sum());yes=int(passing[k].sum())
    per_depth.append(dict(depth_mm=float(depth),evaluated_voxels=count,passing_voxels=yes,
                          failing_voxels=count-yes,pass_rate_percent=100*yes/count if count else None))
cg=next(s.split('::',1)[1] for s in Path('/proc/self/cgroup').read_text().splitlines() if s.startswith('0::'))
cg=Path('/sys/fs/cgroup')/cg.lstrip('/')
cap=int((cg/'memory.max').read_text());assert cap<=30000000000
result=dict(status='COMPLETE',completed_at=datetime.datetime.now().astimezone().isoformat(),
    pymedphys_version=pymedphys.__version__,numpy_version=np.__version__,scipy_version=scipy.__version__,
    reference='TOPAS',evaluation='GPU',histories_per_engine=M['histories_per_engine'],
    distance_criterion_mm=0.,dose_difference_criterion_percent=DOSE_PERCENT,
    normalisation='global TOPAS maximum; no fitted rescaling',lower_reference_dose_cutoff_percent=CUTOFF_PERCENT,
    reference_max_dose_Gy_per_source=normalisation,reference_max_location_mm=dict(depth=float(z[norm_z]),x=float(x[norm_x])),
    absolute_dose_difference_tolerance_Gy_per_source=dose_threshold,
    reference_dose_cutoff_Gy_per_source=cutoff,total_grid_voxels=int(mask.size),
    evaluated_voxels=n,excluded_voxels=int(mask.size-n),passing_voxels=passed,failing_voxels=failed,
    pass_rate_percent=100*passed/n,pass_condition='gamma <= 1; reference dose >= 0.03 * reference maximum',
    gamma_statistics=dict(mean=float(gamma[mask].mean()),median=float(np.median(gamma[mask])),
        p95=float(np.percentile(gamma[mask],95)),p99=float(np.percentile(gamma[mask],99)),maximum=float(gamma[mask].max())),
    calculation_wall_s=elapsed,memory_max_bytes=cap,memory_peak_bytes=int((cg/'memory.peak').read_text()),
    implementation=dict(public_zero_distance_error='PyMedPhys 0.41.0 gamma(..., distance_mm_threshold=0) raises ValueError: Number of samples must be non-negative.',
        method='PyMedPhys GammaInternalFixedOptions + gamma_loop with maximum_test_distance=0; only the same coordinate is evaluated.',
        maximum_test_distance_mm=0.,internal_spatial_denominator_mm=1.,
        spatial_denominator_note='At the only searched distance, zero, the spatial term is exactly zero. The denominator does not permit any displacement.',
        interpolation='scipy RegularGridInterpolator at identical grid coordinates',random_subset=None,
        output_gamma_clipped=False,analytic_formula='abs(GPU - TOPAS) / (0.03 * TOPAS.max())',
        max_abs_difference_vs_analytic=max_diff,classification_mismatches_vs_analytic=0,
        tiny_grid_verification_passed=True,source_sha256=hashlib.sha256(Path(inspect.getsourcefile(shell.gamma_loop)).read_bytes()).hexdigest()),
    scoring_scope='2D x-depth map averaged across the original 100 mm vertical bin; 0.1 mm x and 0.25 mm depth spacing.',
    monte_carlo_uncertainty_note='Pass rate is calculated from the four-batch averaged dose maps; no confidence interval for the pass rate is estimated here.',
    inputs=inputs)
(D/'gamma_summary.json').write_text(json.dumps(result,indent=2)+'\n')
np.savez_compressed(D/'gamma_map.npz',gamma=gamma,evaluated=mask,passing=passing,x_mm=x,depth_mm=z)
with (D/'gamma_by_depth.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(per_depth[0]));w.writeheader();w.writerows(per_depth)

plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,2,figsize=(13,6),constrained_layout=True)
xs=abs(x)<=25;zs=z<=140
extent=[x[xs][0]-.05,x[xs][-1]+.05,z[zs][-1]+.125,0]
im=axs[0].imshow(gamma[zs][:,xs],extent=extent,aspect='auto',vmin=0,vmax=2,cmap='viridis')
axs[0].set(xlabel='Transverse x (mm)',ylabel='Water depth (mm)',title='Gamma map; excluded reference voxels are blank')
fig.colorbar(im,ax=axs[0],label='Gamma (color scale capped at 2)',extend='max' if gamma[mask].max()>2 else 'neither')
axs[1].hist(gamma[mask],bins=100,color='#007d83',alpha=.8)
axs[1].axvline(1,color='#b91c1c',ls='--',label='Passing threshold = 1')
axs[1].set(xlabel='Gamma',ylabel='Evaluated voxel count',title=f'{passed:,} / {n:,} voxels pass ({100*passed/n:.4f}%)',yscale='log');axs[1].legend()
fig.suptitle('PyMedPhys | 0 mm / 3% global | TOPAS reference | 3% reference-dose cutoff\n12.8M histories per engine; zero search radius; exact same-voxel dose comparison')
fig.savefig(D/'gamma_0mm_3pct_cutoff3.png',dpi=170);plt.close(fig)
report=f'''# Gamma：0 mm / 3%，低剂量阈值 3%

通过率：**{100*passed/n:.6f}%**（{passed:,} / {n:,} 个体素通过，{failed:,} 个未通过）。

参考为 TOPAS，评价为 GPU；两端各 1280 万源历史，采用四批平均剂量。全局归一化使用 TOPAS 最大剂量，不进行剂量拟合。

- 距离判据：0 mm，仅比较同一坐标。
- 剂量差判据：TOPAS 全局最大剂量的 3%。
- 低剂量阈值：仅纳入 TOPAS 剂量不低于其最大值 3% 的体素。
- 通过条件：γ ≤ 1；未抽样，计算全部满足剂量阈值的体素。
- TOPAS 最大剂量：{normalisation*1e9:.9f} nGy/源历史。
- 3% 绝对剂量差容限：{dose_threshold*1e9:.9f} nGy/源历史。
- 网格沿用 0.1×100×0.25 mm，结果为沿 100 mm 高度平均后的 x–深度图。

PyMedPhys {pymedphys.__version__} 的公开 gamma 接口不能直接处理零距离。这里使用其 gamma 核心并将最大搜索半径严格设置为 0，空间项恒为零，没有引入非零空间容差。逐体素与解析式 `abs(GPU−TOPAS)/(0.03×TOPAS最大剂量)` 核对：最大差 {max_diff:.3g}，通过/失败分类完全一致。

这个通过率针对全局 3% 剂量差，不表示局部峰/谷剂量误差也小于 3%，也不构成全场 1% 精度证明。本次没有估计蒙卡统计波动对通过率的置信区间。

结果：gamma_summary.json、gamma_map.npz、gamma_by_depth.csv、gamma_0mm_3pct_cutoff3.png。
'''
(D/'GAMMA_RESULT_zh.md').write_text(report)
print(json.dumps({k:result[k] for k in ['status','pymedphys_version','evaluated_voxels','passing_voxels','failing_voxels','pass_rate_percent','reference_max_dose_Gy_per_source','absolute_dose_difference_tolerance_Gy_per_source','calculation_wall_s','memory_peak_bytes','gamma_statistics']},indent=2),flush=True)
