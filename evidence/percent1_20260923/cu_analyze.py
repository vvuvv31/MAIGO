from pathlib import Path
import numpy as np,json,math
from scipy.stats import t as student
r=Path('/mnt/sdb/wuwei/MAIGO_review_fix_20260923');p=r/'evidence/percent1_20260923';o=p/'cu_step025';baseline=json.loads((p/'comparison.json').read_text())
assert len(json.loads((o/'gpu_quality.json').read_text())['runs'])==3
assert all((o/f'topas_1m_s{i}/quality.json').exists() for i in [1,2,3])
g=np.stack([np.fromfile(r/f'out/percent1_cu0025_s{i}/dose.raw',dtype='<f4').reshape(1000,1000).astype(float)/1e6 for i in [1,2,3]])
b=np.stack([np.fromfile(r/f'out/valley_cu0051m_s{i}/dose.raw',dtype='<f4').reshape(1000,1000).astype(float)/1e6 for i in [1,2,3]])
t=np.stack([np.fromfile(o/f'topas_1m_s{i}/dose.bin',dtype='<f8').reshape(1000,1000)/1e6 for i in [1,2,3]])
x=-49.95+np.arange(1000)*.1;z=.125+np.arange(1000)*.25;vm=abs(abs(x)-1.8)<.45;pm=abs(x)<.25
result={'scope':'Cu/slit ceilings halved to 0.025 mm in both engines, water remains 0.05 mm. Three 1M batches each. GPU paired source/seeds against previous three Cu0.05 runs; TOPAS independent seeds. Step study only; not a high-stat replacement for main benchmark.','rows':[]}
for row in baseline['rows']:
 depth=row['depth_mm'];width=row['window_width_mm'];mask=np.arange(1000)==int(np.argmin(abs(z-depth))) if width==.25 else abs(z-depth)<width/2
 rr={'depth_mm':depth,'window_width_mm':width}
 for name,m in [('valley',vm),('peak',pm),('row_mean',np.ones(1000,bool))]:
  a=g[:,mask][:,:,m].mean((1,2));a0=b[:,mask][:,:,m].mean((1,2));ref=t[:,mask][:,:,m].mean((1,2));oldref=row[name]['topas']['mean'];dg=(a-a0)/oldref
  ratio=a.mean()/ref.mean();vg=a.var(ddof=1)/3/ref.mean()**2;vt=ratio**2*ref.var(ddof=1)/3/ref.mean()**2;se=math.sqrt(vg+vt);df=(vg+vt)**2/(vg*vg/2+vt*vt/2);ci=float(student.ppf(.975,df))*se
  # New TOPAS versus old Cu0.05 pooled estimate, independent batches.
  dt=ref.mean()/oldref-1;dts=math.sqrt(ref.var(ddof=1)/3+(ref.mean()/oldref*row[name]['topas']['SE'])**2)/oldref
  rr[name]={'GPU_paired_step_change':float(dg.mean()),'GPU_paired_step_change_SE':float(dg.std(ddof=1)/math.sqrt(3)),'TOPAS_step_change':float(dt),'TOPAS_step_change_SE':float(dts),'GPU_vs_TOPAS_relative_difference':float(ratio-1),'combined_SE':se,'CI95_approx':[float(ratio-1-ci),float(ratio-1+ci)],'GPU_values':a.tolist(),'TOPAS_values':ref.tolist(),'GPU_15M_baseline_plus_paired_shift_vs_new_TOPAS':float((row[name]['gpu']['mean']+(a-a0).mean())/ref.mean()-1)}
 result['rows'].append(rr)
 v=rr['valley'];print(depth,width,'GPU shift pp',100*v['GPU_paired_step_change'],'GPU shift SE',100*v['GPU_paired_step_change_SE'],'TOPAS shift pp',100*v['TOPAS_step_change'],'matched new diff%',100*v['GPU_vs_TOPAS_relative_difference'],'CI95%',[100*v for v in v['CI95_approx']],flush=True)
(o/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['| 深度 mm | GPU减半变化 pp ± 1SE | TOPAS减半变化 pp ± 1SE | 0.025 mm两端谷区差 | 近似95%区间 |','|---:|---:|---:|---:|---:|']
for row in result['rows']:
 if row['window_width_mm']!=.25:continue
 v=row['valley'];ci=v['CI95_approx'];lines.append(f"| {row['depth_mm']} | {100*v['GPU_paired_step_change']:+.2f} ± {100*v['GPU_paired_step_change_SE']:.2f} | {100*v['TOPAS_step_change']:+.2f} ± {100*v['TOPAS_step_change_SE']:.2f} | {100*v['GPU_vs_TOPAS_relative_difference']:+.2f}% | [{100*ci[0]:+.2f}%, {100*ci[1]:+.2f}%] |")
(o/'RESULTS.md').write_text('# 铜段步长减半检查\n\n'+result['scope']+'\n\n'+ '\n'.join(lines)+'\n\n不同TOPAS种子下的变化仍含统计噪声。GPU配对能降低一部分源抽样方差，但改变步数后散射随机数对应关系也会变化。只用此表的点估计不能把0.025 mm宣称为1%解决方案。`GPU_15M_baseline_plus_paired_shift_vs_new_TOPAS`仅为方差降低的估计值，未冒充实际15M的0.025 mm运行。\n')
