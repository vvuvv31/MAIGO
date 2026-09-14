from pathlib import Path
import json,csv,re,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;B=R/'benchmarks';P=B/'plots';old=R.parent/'delta_moments_review';prod=R.parents[1]/'benchmark20260914/production_continuation';rows=[];layers=[]
def r80(z,v):
 p=v.argmax();i=p+np.flatnonzero(v[p:]<.8*v[p])[0];return float(z[i-1]+(.8*v[p]-v[i-1])*(z[i]-z[i-1])/(v[i]-v[i-1]))
fig,axes=plt.subplots(5,2,figsize=(12,17),constrained_layout=True)
for ax,(name,*_) in zip(axes.ravel(),json.loads((B/'cases.json').read_text())):
 d=B/name;status=json.loads((d/'gpu_status.json').read_text());assert status['complete'] and status['quality']['queue_overflow_count']==0
 assert all(a[0]==a[3]==a[4]==0 for a in status['audit'])
 z,t,g=np.loadtxt(d/'idd.csv',delimiter=',').T;_,_,o=np.loadtxt(old/'benchmarks'/name/'idd.csv',delimiter=',').T;_,_,p=np.loadtxt(prod/name/'idd.csv',delimiter=',').T
 c=json.loads((d/'comparison.json').read_text());assert abs(c['gpu_voxel_energy_to_ledger_ratio']-1)<1e-4
 use=t>.1*t.max();row=dict(case=name,peak_error_percent=100*(g.max()/t.max()-1),old_peak_error_percent=100*(o.max()/t.max()-1),r80_shift_mm=r80(z,g)-r80(z,t),old_r80_shift_mm=r80(z,o)-r80(z,t),mare_percent=float(np.mean(abs(g[use]/t[use]-1))*100),old_mare_percent=float(np.mean(abs(o[use]/t[use]-1))*100),integral_ratio=g.sum()/t.sum());rows.append(row)
 limit=min(z[-1]+.25,1.2*z[t.argmax()])
 for v,label,style in [(t,'TOPAS','-'),(p,'GPU production','--'),(o,'GPU moments',':'),(g,'GPU along-step','-')]:ax.plot(z,v/t.max(),label=label,linestyle=style,lw=1.1)
 ax.set(xlim=(0,limit),title=name,xlabel='Depth (mm)',ylabel='IDD / TOPAS peak');ax.grid(alpha=.2)
 f,aa=plt.subplots(2,1,figsize=(9,7),sharex=True)
 for v,label in [(p,'GPU production'),(o,'GPU moments'),(g,'GPU along-step')]:
  use=(z<=limit)&(t>.01*t.max());err=np.full_like(t,np.nan);err[use]=100*(v[use]/t[use]-1);aa[0].plot(z,err,label=label);aa[1].plot(z,100*(v-t)/t.max(),label=label)
 for a in aa:a.set_xlim(0,limit);a.legend();a.grid(alpha=.2);a.axhline(0,color='gray',lw=.6)
 aa[0].set(title=name+' IDD error (unrestricted y)',ylabel='Relative error (%)\nTOPAS > 1% peak');aa[1].set(xlabel='Depth (mm)',ylabel='Difference / TOPAS peak (%)');f.tight_layout()
 for ext in ['png','pdf']:f.savefig(P/f'{name}_idd_relative_error_full.{ext}',dpi=160)
 plt.close(f)
 if name in ['b1_100','b1_200','b1_300']:
  i=t.argmax()
  for j in range(i-2,i+3):layers.append(dict(case=name,depth_mm=z[j],topas_peak_fraction=t[j]/t.max(),old_local_error_percent=100*(o[j]/t[j]-1),new_local_error_percent=100*(g[j]/t[j]-1),production_local_error_percent=100*(p[j]/t[j]-1)))
axes[0,0].legend();fig.suptitle('Along-step mean correction; same aggregate delta sampler')
for ext in ['png','pdf']:fig.savefig(P/f'b1_b4_idd_overview.{ext}',dpi=170)
plt.close(fig)
for name,items in [('summary',rows),('peak_layers',layers)]:
 (B/f'{name}.json').write_text(json.dumps(items,indent=2))
 with (B/f'{name}.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=items[0]);w.writeheader();w.writerows(items)
s=json.loads((R/'shard_results.json').read_text());o=json.loads((old/'shard_results.json').read_text())
text=f'''# 沿步均值修正候选

基于delta_moments_review隔离候选；未修改生产代码/配置/二进制，未commit/push。全部18种已支持带电离子共用。

## 改动与限制

- 在已定位的restricted stopping三次多项式区间解析计算导数，不额外搜索表。
- 仅在线性基本能损分支使用 S_eff = S0 − h/2*(S0+D0)*dS0/dT。原range反演分支保留，避免重复修正。
- 原生低能/电荷修正保留，其预测中点再考虑δ平均损失。δ均值也用同一区间M1导数沿步修正。
- δ方差仍在步首取值；Gamma抽样器/原restricted涨落不变，不新增δ碰撞距离限制，不改步长配置。
- 该实现是二阶局部近似；不是新的完整总range反演，不保证跨表格模型边界时二阶精度。非正/越过原linear_limit的restricted修正回到原分支；δ均值取非负。
- 复用原EM与EMDMOMT2矩表，原Schneider核数据保持v2.1；运行前执行验证。
- C++测试验证三次多项式导数、端点外常值、解析线性stopping的二阶收敛及δ均值耦合。

## RT07575 1/20 shard

6,481,909原发，仍使用两个相同spot权重子任务及seed；不启用独立核弹性。下表wall为两个进程之和，包含加载/输出，不含排队，合并另计；运行零overflow。

|方案|完整wall s|程序Elapsed s|程序histories/s|primary s|secondary s|
|---|---:|---:|---:|---:|---:|
|原两矩Gamma|{o['total_wall_s']:.3f}|{o['total_elapsed_s']:.3f}|{o['throughput_hps']:.1f}|{o['primary_s']:.3f}|{o['secondary_s']:.3f}|
|沿步修正|{s['total_wall_s']:.3f}|{s['total_elapsed_s']:.3f}|{s['throughput_hps']:.1f}|{s['primary_s']:.3f}|{s['secondary_s']:.3f}|

## b1–b4

各20万原发，固定原TOPAS参考，核弹性保持原benchmark设置。IDD从3D横向求和。全部运行零overflow、EM查表无错误，3D积分与能量账本一致。

[IDD总览](benchmarks/plots/b1_b4_idd_overview.png) · [全部图表](benchmarks/plots/README.md)

|Case|原峰误差 %|新峰误差 %|原R80偏移 mm|新R80偏移 mm|原MARE %|新MARE %|
|---|---:|---:|---:|---:|---:|---:|
'''
for v in rows:text+=f"|{v['case']}|{v['old_peak_error_percent']:+.3f}|{v['peak_error_percent']:+.3f}|{v['old_r80_shift_mm']:+.3f}|{v['r80_shift_mm']:+.3f}|{v['old_mare_percent']:.3f}|{v['mare_percent']:.3f}|\n"
text+='\n峰误差为最大IDD比，R80为远端各自峰高80%位置差，MARE取TOPAS>10%峰值层。不能用峰高代替逐层误差。\n\n|Case|深度 mm|TOPAS/峰值|原逐层误差 %|新逐层误差 %|生产逐层误差 %|\n|---|---:|---:|---:|---:|---:|\n'
for v in layers:text+=f"|{v['case']}|{v['depth_mm']:.2f}|{v['topas_peak_fraction']:.3f}|{v['old_local_error_percent']:+.2f}|{v['new_local_error_percent']:+.2f}|{v['production_local_error_percent']:+.2f}|\n"
text+='\n单次对照，未评估重复运行波动；患者BODY Gamma尚未计算。候选未验收为生产。\n'
text+='\n## 结果判断\n\nRT07575为66.54秒、102.7k histories/s；比未修正两矩候选多1.07秒，单次测量吞吐下降约1.7%，仍满足100秒目标。\n100/200/300 MeV/u的高剂量层MARE由0.93/1.36/1.06%降为0.39/0.54/0.50%，R80偏移降为−0.002/−0.011/−0.036 mm，说明沿步积分是偏差的重要来源。\n但是零能散峰高变为−2.94/−3.64/−1.64%，200 MeV/u峰后一层仍低19.38%，不能认定峰区问题已完全修复。\n可保留为下一步研究基线，后续优先检查三矩合并分布及其全部材料适用条件；不建议凭单组结果人为缩放峰值或拟合能量相关修正系数。尚未将剩余偏差归因到单一机制。\n'
(R/'README.md').write_text(text);print(text)
