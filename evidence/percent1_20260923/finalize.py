from pathlib import Path
import json,hashlib,tarfile,subprocess,shutil
r=Path('/mnt/sdb/wuwei/MAIGO_review_fix_20260923');o=r/'evidence/percent1_20260923';cu=o/'cu_step025'
j=json.loads((o/'comparison.json').read_text());c=json.loads((cu/'comparison.json').read_text());q=json.loads((o/'delta_fixed_quality.json').read_text())
assert j['status']=='COMPLETE' and len(q['runs'])==7
summary='''\n\n## 铜段步长收敛检查完成

额外完成GPU/TOPAS各3个种子、每种子100万历史，将准直器Cu及空气狭缝的步长上限都从0.05减到0.025 mm，水仍为0.05 mm。100 mm单层谷区相对于原TOPAS剂量，GPU配对变化−4.24±1.49个百分点（1SE），TOPAS独立批次变化−3.12个百分点；10 mm窗口分别−3.11±0.62和−3.03个百分点。两端都显示步长依赖，因此不能只把GPU降低后的数值当成修正成功。

0.025 mm两端直接比较的六深度单层谷区差为 −0.53%、−1.18%、−0.38%、+3.63%、+1.19%、+1.99%，统计区间明显更宽。此设置未达到全深度1%目标，未替换0.05 mm主配置。详情见 `cu_step025/RESULTS.md`。确定性Cu平均能损积分在20 mm铜后，步长0.05→0.025只使出射能量约47.15 MeV/u变化0.00163 MeV/u，不能解释完整模拟中的数个百分点变化。

本轮已完成：独立种子统计、入口相空间比较、电子迁移评分缺陷修复及回归、两端Cu步长检查。**1%整体验收尚未达成**。后续应在更稳定的Cu步长/共同出口相空间上下文中隔离散射、几何边界及能损涨落，再做高统计验收；不应以整体归一化或修改Urban幅度来压低当前结果。

代码修复和过程分解见 `PROCESS_DIAGNOSIS.md`；主统计图 `statistics.png`，粒子数累计轨迹 `convergence.png`，入口图 `entrance_comparison.png`。
'''
p=o/'RESULTS.md';s=p.read_text()
if '## 铜段步长收敛检查完成' not in s:p.write_text(s+summary)
p=o/'PROCESS_DIAGNOSIS.md';s=p.read_text().replace('目前正在检查Cu0.05→0.025 mm的独立TOPAS和配对GPU收敛，结果存于 `cu_step025/`。','Cu0.05→0.025 mm的独立TOPAS和配对GPU收敛检查已完成，见 `cu_step025/RESULTS.md`；两端都有谷区步长响应，未达到全深度1%目标，未据此调参。');p.write_text(s)
for filename,marker,body in [('urban.md','## 2026-09-23 Cu/slit 0.025 mm convergence check','Both engines were rerun at Cu/slit 0.025 mm, water 0.05 mm (3 x 1M each). At 100 mm, GPU paired valley shift is -4.24 +/-1.49 pp, TOPAS shift -3.12 pp; their 10 mm-window shifts are -3.11 and -3.03 pp. The finer matched result has wider statistics and does not establish 1% across depths. Keep the 0.05 mm main preset; no coefficient tuning or default promotion. Evidence: evidence/percent1_20260923/cu_step025/RESULTS.md.'),('failed.md','### 2026-09-23 Claiming 1% convergence by halving only the GPU Cu step','Both engines respond when Cu/slit ceilings go 0.05 -> 0.025 mm. At 100 mm their 10 mm-window valleys both drop about 3.1 pp relative to the original reference. The finer matched single-layer result remains +3.63% at 80 mm and +1.99% at 120 mm with broad three-batch intervals. Do not compare finer GPU to unchanged TOPAS or select one favorable depth as a convergence proof. More independent statistics and a common-phase-space / boundary-step study are needed. Evidence: evidence/percent1_20260923/cu_step025/RESULTS.md.')]:
 p=r/filename;s=p.read_text()
 if marker not in s:p.write_text(s+'\n\n'+marker+'\n\n'+body+'\n')
# Save the tested opt-in diagnostic without changing the main preset.
preset=r/'config/review_fullchain_delta_response_diagnostic_1m.yaml'
if not preset.exists():preset.write_text('# Research diagnostic; not a <1% valley fix or default promotion.\n'+(r/'config/percent1_delta_fixed_s2.yaml').read_text())
subprocess.run(['git','diff','--check'],cwd=r,check=True)
cache=(r/'build/oneapi-nvidia-minibeam/CMakeCache.txt').read_text()
for line in ['CARBON_DOSE_FP32:BOOL=ON','CARBON_DOSE_FP64:BOOL=OFF','CARBON_DISABLE_INTEGRITY_CHECKS:BOOL=OFF']:assert line in cache,line
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
assert sha(r/'build/oneapi-nvidia-minibeam/carbon_mc')==q['fixed_binary_sha256']
files={}
for p in o.rglob('*'):
 if p.is_file() and p.suffix in ['.py','.json','.md','.log','.png','.patch','.txt','.yaml','.binheader','.header','.csv'] and p.name!='artifact_manifest.json':
  files[p]=str(p.relative_to(o))
for p in o.glob('*.csv'):files[p]=p.name
for p in r.glob('config/percent1_*.yaml'):files[p]='config/'+p.name
for p in [r/'config/review_fullchain_topas_matched_1m.yaml',preset]:files[p]='config/'+p.name
for p in [r/'urban.md',r/'failed.md',r/'AGENTS.md']:files[p]='repository_notes/'+p.name
files[r/'src/transport_sycl.cpp']='source/transport_sycl.cpp';files[o/'transport_sycl.before_delta_fix.cpp']='source/transport_sycl.before_delta_fix.cpp'
external={r/'build/oneapi-nvidia-minibeam/carbon_mc',o/'carbon_mc_before_delta_fix'}
for p in o.rglob('*'):
 if p.is_file() and p.suffix in ['.bin','.phsp']:external.add(p)
for name in [f'percent1_matched_s{i}' for i in range(1,13)]+[f'valley_cu0051m_s{i}' for i in [1,2,3]]+list(q['runs'])+[f'percent1_cu0025_s{i}' for i in [1,2,3]]+['percent1_delta_tail_smoke']:
 for fn in ['dose.raw','dose.mhd','quality_report.json','energy_ledger.json']:
  p=r/f'out/{name}/{fn}'
  if p.exists():external.add(p)
ref=Path('/mnt/sda/wuwei/minibeam_single_center_em_only_water75ev_e250_10m/topas_7378')
external.update(p for p in ref.iterdir() if p.is_file())
external.add(r/'evidence/valley_diagnosis_20260923/entrance_cu005_s1.csv')
external.update([r/'evidence/valley_diagnosis_20260923/source_1m.csv',r/'evidence/review_fix_20260923/active_water005.csv',r/'evidence/review_fix_20260923/active_copper005.csv',Path('/mnt/sda/wuwei/minibeam_copper_extract_e250/compiled/c12_copper_stopping_geant4_11_3_2.csv')])
external.update(Path('/mnt/sda/wuwei/water75ev_c12_response_20260922').glob('table*'))
manifest={'worktree':str(r),'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=r,text=True).strip(),'goal_1pct_certified':False,'old_binary_sha256':q['old_binary_sha256'],'fixed_binary_sha256':q['fixed_binary_sha256'],'FP32_dose':True,'FP64_dose':False,'integrity_checks':True,'files':[{'path':str(p),'archive':arc,'size':p.stat().st_size,'sha256':sha(p)} for p,arc in sorted(files.items(),key=lambda v:v[1])],'external_evidence':[{'path':str(p),'size':p.stat().st_size,'sha256':sha(p)} for p in sorted(external)]}
(o/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');files[o/'artifact_manifest.json']='artifact_manifest.json'
archive=o/'percent1_review_bundle.tar.gz'
with tarfile.open(archive,'w:gz') as tf:
 for p,arc in sorted(files.items(),key=lambda v:v[1]):tf.add(p,arcname=arc)
with tarfile.open(archive,'r:gz') as tf:assert len(tf.getmembers())==len(files)
print('FINAL',json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),'members':len(files),'external':len(external),'fixed_binary_sha256':q['fixed_binary_sha256']}),flush=True)
