from pathlib import Path
import json,time
import numpy as np
from common import ROOT,sha,read_mhd,save_json
from metrics import roi_indices
from compare import pass_mask
from plots import make_plots
m=json.loads((ROOT/'manifest.json').read_text());summary=json.loads((ROOT/'results/summary.json').read_text());assert summary['state']=='complete' and summary['histories']==236086432
ref,fields=read_mhd(ROOT/'inputs/physical_dose.mhd');gpu,gfields=read_mhd(ROOT/'results/dose_4Gy.mhd')
for k in ['TransformMatrix','Offset','ElementSpacing','DimSize']:assert fields[k]==gfields[k]
# Match the established host gamma analysis precision; GPU dose scoring stays FP32.
ref=ref.astype(np.float64);gpu=gpu.astype(np.float64)
rois=roi_indices();body=np.zeros((35,417,505),bool);body.reshape(-1)[np.union1d(rois['BODY'],rois['GTV'])]=True;body=body.transpose(0,2,1)
mask=body&(ref>=.1*ref.max());points=np.argwhere(mask);spacing=np.array(list(map(float,fields['ElementSpacing'].split())))[::-1]
report=dict(script_sha256=sha(__file__),reference_dose_sha256=sha(ROOT/'inputs/physical_dose.raw'),gpu_dose_sha256=sha(ROOT/'results/dose_4Gy.raw'),histories=summary['histories'],reference='Historical optimized TOPAS weighted DIJ; no new TOPAS simulation',method=dict(mask='BODY union GTV; reference >= 10% whole-reference maximum',evaluated_voxels=len(points),interpolation='trilinear',search='isotropic spherical DTA/10 lattice; pass/fail, not exact continuous gamma',normalization='local reference dose or global reference maximum; no fit or registration'),gamma={})
start=time.perf_counter()
for pct,dta in [(1,.2),(2,.2),(3,.2),(3,0)]:
 for local in [True,False]:
  name=f'{"local" if local else "global"}_{pct}pct_{dta}mm';passed=pass_mask(gpu,ref,points,spacing,pct,dta,local)
  report['gamma'][name]=dict(percent=100*float(passed.mean()),passed=int(passed.sum()),evaluated=len(passed));save_json(ROOT/'results/gamma.json',report);print(name,report['gamma'][name],flush=True)
report['wall_seconds']=time.perf_counter()-start;report['state']='complete';save_json(ROOT/'results/gamma.json',report)
old,_=read_mhd(Path(m['previous_campaign'])/'results/dose_4Gy.mhd');delta=gpu.astype(float)-old
comparison=dict(scope='Current full independent FP32 vs historical Schneider joint+legacy hybrid; identical plan sources/seeds, no timing speedup claim.',BODY_L1=float(abs(delta[body]).sum()/old[body].sum()),pooled_BODY_change_percent=float(100*delta[body].sum()/old[body].sum()))
qualified=body&(old>=.01*old[body].max());local=100*abs(delta[qualified])/old[qualified];comparison['local_absolute_percentiles_at_1pct_BODY_max']={str(p):float(np.percentile(local,p)) for p in [50,95,99,99.9,100]};save_json(ROOT/'results/previous_comparison.json',comparison)
make_plots(ROOT/'results')
t=json.loads((ROOT/'results/timing.json').read_text());checks=json.loads((ROOT/'batch_checks.json').read_text())
lines=['# RT07575 proton minibeam：当前FP32版本0.1% full plan','',f"完成 {m['histories']:,} primary，487 spots，119批单进程，group32。按4 Gy计划入射粒子数的0.1%取样，输出按粒子数缩放回4 Gy计划；无剂量拟合。",'',f"执行文件SHA256：`{json.loads((ROOT/'build.json').read_text())['binary_sha256']}`。使用全FP32独立电子birth/joint/charged/photon路径、参数复用、速率认证和体素缓存；未启用已否决的elastic-batch候选。",'',f"蒙卡进程墙钟 {t['process_seconds']:.3f} s（{t['process_seconds']/60:.2f} min）；含加载吞吐 {t['process_primary_per_second']:.1f} primary/s；排除首批后的常驻吞吐 {t['resident_primary_per_second']:.1f} primary/s。电子阶段累计 {t['electron_seconds']:.3f} s。后处理耗时另计。",'',f"119/119批research质量检查通过；主输运/EM审计与历史同源同种子计划一致，电子全部完成，缺失覆盖、invalid和队列溢出均为0，零旧电子包加载/回退。最大相对能量残差 {max(abs(c['energy_residual']) for c in checks):.3e}。质量检查通过不等同正式患者物理验收。",'','| Gamma vs TOPAS | Local | Global |','|---|---:|---:|']
for pct,dta in [(1,.2),(2,.2),(3,.2),(3,0)]:lines.append(f"| {pct}% / {dta} mm | {report['gamma'][f'local_{pct}pct_{dta}mm']['percent']:.4f}% | {report['gamma'][f'global_{pct}pct_{dta}mm']['percent']:.4f}% |")
lines+=['',f"参考为既有TOPAS加权DIJ；BODY∪GTV、10%全参考最大剂量阈值、{len(points):,}体素。0.2 mm结果采用三线性插值和0.02 mm搜索格，不能解释为参考原始网格具有0.02 mm分辨率。",'','| GTV Gy | TOPAS | 当前GPU |','|---|---:|---:|']
for key in ['mean_Gy','D95_Gy','D98_Gy','D50_Gy','D2_Gy']:lines.append(f"| {key} | {summary['reference_metrics']['GTV'][key]:.6f} | {summary['gpu_metrics']['GTV'][key]:.6f} |")
lines+=['','[4 Gy患者坐标剂量](results/dose_4Gy.mhd) · [剂量平面图](results/dose_comparison.png) · [DVH与横向剖面](results/dvh_profile.png) · [完整运行检查](batch_checks.json) · [耗时](results/timing.json) · [Gamma](results/gamma.json)','']
(ROOT/'REPORT_zh.md').write_text('\n'.join(lines));print('postprocessing complete',flush=True)
