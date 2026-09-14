from pathlib import Path
import json,re,numpy as np
root=Path(__file__).resolve().parent;repo=root.parents[1];r=repo/'scratch/unified_em_perf_20260913'
b=r/'runtime_current_1m_baseline/RT07575';d=r/'runtime_current_1m_phase_deep/RT07575'
sb=json.loads((b/'status.json').read_text());sd=json.loads((d/'status.json').read_text())
assert sd['complete'] and sd['quality']['queue_overflow_count']==0 and sb['audit']==sd['audit'] and sb['steps']==sd['steps'] and sb['config_sha256']==sd['config_sha256']
x=np.fromfile(b/'dose_gpu.raw','<f4').astype(float);y=np.fromfile(d/'dose_gpu.raw','<f4').astype(float);diff=100*abs(y-x).max()/x.max();assert diff<.001
la=json.loads((b/'out/gpu/energy_ledger.json').read_text());lb=json.loads((d/'out/gpu/energy_ledger.json').read_text());events={k:[v,lb[k]] for k,v in la.items() if isinstance(v,int) and 'interactions' in k};assert all(a==b for a,b in events.values())
c={int(i):[int(t),int(n)] for i,t,n in re.findall(r'\[phase-cycles\] (\d+) (\d+) (\d+)',(d/'gpu.log').read_text())}
groups=[('Primary',[(0,'CT/material and initial state'),(6,'EM material selection/coverage'),(26,'EM stopping/range/fluctuation parameter preparation'),(27,'Delta rate/clock/EM step'),(7,'Geometry clamps'),(24,'Nuclear rate/hazard'),(25,'Pre-loss mean/audit/legacy branches'),(1,'Unified EM loss'),(2,'Scoring branches'),(3,'MCS'),(4,'Advance/face handling'),(5,'Nuclear resolution/bookkeeping')]),('Secondary',[(8,'CT/material/stopping and initial state'),(14,'EM material selection/coverage'),(28,'EM stopping/range/fluctuation parameter preparation'),(29,'Delta rate/clock/EM step'),(15,'Geometry clamps'),(30,'Nuclear rate/hazard'),(31,'Other pre-loss/recoil/legacy branches'),(9,'Unified EM loss'),(10,'Post-EM geometry/scoring'),(11,'MCS'),(12,'Inelastic replay/scoring'),(13,'Elastic/bookkeeping')])]
lines=['# RT07575 preparation breakdown — deep probe','','沿用 runtime_current_1m_baseline 的冻结配置、粒子数、随机种子和物理数据。正式源码未插桩。','',f'诊断 Elapsed={sd["elapsed_s"]:.3f} s；相对基准增加 {100*(sd["elapsed_s"]/sb["elapsed_s"]-1):.2f}%。最大体素剂量差={diff:.8f}% of peak；步数、EM audit、核反应计数相同，零 overflow。','','以下为每约 2048 步采样 lane elapsed cycles 的分布，含等待/分歧/插桩效应，不是各过程独立 wall-time；不能直接换算可消除秒数。','']
rows={}
for name,slots in groups:
 total=sum(c[i][0] for i,label in slots);rows[name]=[dict(slot=i,region=label,cycles=c[i][0],intervals=c[i][1],share_percent=100*c[i][0]/total) for i,label in slots]
 lines += [f'## {name}','','| Region | Sampled cycle share | Intervals |','|---|---:|---:|']
 for row in rows[name]:lines.append(f'| {row["region"]} | {row["share_percent"]:.2f}% | {row["intervals"]} |')
 lines.append('')
(root/'deep_results.json').write_text(json.dumps(dict(status=sd,cycles=c,regions=rows,max_dose_delta_percent_peak=diff,event_counts=events),indent=2)+'\n')
(root/'DEEP.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axs=plt.subplots(1,2,figsize=(15,7),constrained_layout=True)
for ax,(name,rs) in zip(axs,rows.items()):
 rs=sorted(rs,key=lambda z:-z['share_percent']);ax.barh([z['region'] for z in rs],[z['share_percent'] for z in rs]);ax.invert_yaxis();ax.set_title(name);ax.set_xlabel('Sampled lane cycle share (%) — not wall-time')
 for j,z in enumerate(rs):ax.text(z['share_percent']+.3,j,f'{z["share_percent"]:.1f}%',va='center',fontsize=8)
 ax.set_xlim(0,max(z['share_percent'] for z in rs)*1.17)
fig.savefig(root/'deep_breakdown.png',dpi=150)
