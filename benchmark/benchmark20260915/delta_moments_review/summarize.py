from pathlib import Path
import json,csv,re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;B=R/'benchmarks';base=R.parents[1]/'benchmark20260914/production_continuation';off=R.parents[1]/'benchmark20260914/delta_off_review/benchmarks';rows=[]
def r80(z,v):
 p=v.argmax();idx=np.flatnonzero(v[p:]<.8*v[p]);assert len(idx);i=p+idx[0];return float(z[i-1]+(.8*v[p]-v[i-1])*(z[i]-z[i-1])/(v[i]-v[i-1]))
for n,e,spread in json.loads((B/'cases.json').read_text()):
 d=B/n;s=json.loads((d/'gpu_status.json').read_text());assert s['complete'] and s['quality']['queue_overflow_count']==0;assert all(a[3]==a[4]==0 and a[6]>0 for a in s['audit'])
 c=json.loads((d/'comparison.json').read_text());z,t,g=np.loadtxt(d/'idd.csv',delimiter=',').T;_,_,p=np.loadtxt(base/n/'idd.csv',delimiter=',').T;_,_,o=np.loadtxt(off/n/'idd.csv',delimiter=',').T;log=(d/'gpu.log').read_text();assert '[delta-moments]' in log
 hps=float(re.search(r'^Throughput: ([\d.]+)',log,re.M)[1]);elapsed=float(re.search(r'^Elapsed: ([\d.]+)',log,re.M)[1])
 rows.append(dict(case=n,energy_MeVu=e,histories=200000,peak_error_percent=c['peak_error_percent'],baseline_peak_error_percent=float(100*(p.max()/t.max()-1)),r80_shift_mm=r80(z,g)-r80(z,t),delta_off_r80_shift_mm=r80(z,o)-r80(z,t),baseline_r80_shift_mm=r80(z,p)-r80(z,t),integral_ratio=c['energy_ratio_GPU_TOPAS'],mare_percent=c['high_dose_mare_percent'],throughput_hps=hps,elapsed_s=elapsed))
 limit=min(z[-1]+.25,1.2*z[t.argmax()]);use=(z<=limit)&(t>.01*t.max());fig,ax=plt.subplots(2,1,figsize=(9,7),sharex=True)
 for v,label in [(p,'GPU production'),(g,'GPU delta moments')]:
  rel=np.full_like(t,np.nan);rel[use]=100*(v[use]/t[use]-1);ax[0].plot(z,rel,label=label);ax[1].plot(z,100*(v-t)/t.max(),label=label)
 ax[0].set_title(n+' IDD error: unrestricted y range');ax[0].set_ylabel('Relative error (%)\nTOPAS > 1% peak');ax[1].set_ylabel('Difference / TOPAS peak (%)');ax[1].set_xlabel('Depth (mm)')
 for a in ax:a.set_xlim(0,limit);a.legend();a.grid(alpha=.2);a.axhline(0,color='gray',lw=.6)
 fig.tight_layout()
 for ext in ['png','pdf']:fig.savefig(B/'plots'/f'{n}_idd_relative_error_full.{ext}',dpi=160)
 plt.close(fig)
(B/'summary.json').write_text(json.dumps(rows,indent=2))
with (B/'summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
fig,axs=plt.subplots(5,2,figsize=(12,17),constrained_layout=True)
for ax,(n,e,spread) in zip(axs.ravel(),json.loads((B/'cases.json').read_text())):
 z,t,g=np.loadtxt(B/n/'idd.csv',delimiter=',').T;_,_,p=np.loadtxt(base/n/'idd.csv',delimiter=',').T
 for v,label,style in [(t,'TOPAS','-'),(p,'GPU production','--'),(g,'GPU delta moments','-')]:ax.plot(z,v/t.max(),label=label,linestyle=style,lw=1.2)
 ax.set(xlim=(0,min(350,1.2*z[t.argmax()])),title=n,xlabel='Depth (mm)',ylabel='IDD / TOPAS peak');ax.grid(alpha=.2)
axs[0,0].legend();fig.suptitle('Aggregate delta loss: Gamma matched to mean and variance')
for ext in ['png','pdf']:fig.savefig(B/'plots'/f'b1_b4_idd_overview.{ext}',dpi=170)
plt.close(fig)

s=json.loads((R/'shard_results.json').read_text());prior=R.parents[1]/'benchmark20260914';mean=json.loads((prior/'delta_mean_review/shard_results.json').read_text());prod=json.loads((prior/'rt07575_shard01_continuation/results.json').read_text())['status'];meanrows={r['case']:r for r in json.loads((prior/'delta_mean_review/benchmarks/summary.json').read_text())}
text=f"""# 第二阶段：每步一次δ合并能损抽样

隔离候选，基于 a7c382d 的均值补偿实验；原发及全部18种已支持带电离子启用。正式源码、配置、二进制未修改，未commit/push。

## 模型

- 不使用逐次δ碰撞抽样、δ时钟或δ碰撞距离限制。
- 对原生proposal rate乘接受后的谱积分，提取 M1=lambda*E[epsilon*accept]、M2=lambda*E[epsilon²*accept]；包含原spin/form-factor/magnetic veto。
- 每步长度h，μ=h*M1，v=h*M2。v是复合Poisson总能损方差，不能使用h*lambda*(E[epsilon²]−E[epsilon]²)。
- Gamma形状k=μ²/v、尺度theta=v/μ，每步抽取一次总δ能损；内部使用Marsaglia–Tsang rejection，通常需要多个uniform，并非一次RNG调用。
- 保留原restricted均值/涨落、核反应、MCS、几何、1%合并平均能损步长上限、64步续跑。
- 在同一步扣动能并局部沉积；按剩余动能截断。截断会改变理论矩，未单独计量其频率。
- Gamma只匹配前两矩，不精确复现零碰撞概率、单次转移上限和高阶谱形；这是一项近似实验。
- 原EM数据未修改。派生EMDMOMT2表206,977,416字节，SHA/原包绑定见moments_table_manifest.json。
- 两矩共31648个节点的32/64点积分复核最大相对误差1.02e-7；实际C++抽样器7组shape，每组200万次，均值/方差统计测试通过（sampler_test.log）。这不等于剂量精度验收。

## RT07575 1/20 shard：6,481,909原发

沿用均值/δOFF的两个spot权重子任务和seed，粒子数精确一致。RT07575配置不启用独立核弹性；b1–b4保留原全离子弹性。
完整耗时为两个成功进程wall之和，不含锁等待，包含加载和输出；合并另计。患者BODY Gamma尚未计算。

|方案|完整wall s|程序Elapsed s|程序histories/s|primary s|secondary s|
|---|---:|---:|---:|---:|---:|
|生产|{prod['wall_s']:.3f}|{prod['elapsed_s']:.3f}|{prod['throughput']:.1f}|{prod['primary_s']:.3f}|{prod['secondary_s']:.3f}|
|δ均值|{mean['total_wall_s']:.3f}|{mean['total_elapsed_s']:.3f}|{mean['throughput_hps']:.1f}|{mean['primary_s']:.3f}|{mean['secondary_s']:.3f}|
|δ均值+方差|{s['total_wall_s']:.3f}|{s['total_elapsed_s']:.3f}|{s['throughput_hps']:.1f}|{s['primary_s']:.3f}|{s['secondary_s']:.3f}|

全部零overflow、运行质量通过，合并剂量为RT07575_delta_moments_merged.mhd/raw。运行质量通过不等于生产精度验收。

## b1–b4：每例200,000原发，共10例

固定TOPAS参考，没有改动TOPAS参数。全部零overflow，EM查表无错误。
[IDD总览](benchmarks/plots/b1_b4_idd_overview.png) · [全部图](benchmarks/plots/README.md)

|Case|均值版峰误差 %|均值+方差峰误差 %|生产峰误差 %|均值+方差R80偏移 mm|积分比|MARE %|
|---|---:|---:|---:|---:|---:|---:|
"""
for x in rows:text+=f"|{x['case']}|{meanrows[x['case']]['peak_error_percent']:+.3f}|{x['peak_error_percent']:+.3f}|{x['baseline_peak_error_percent']:+.3f}|{x['r80_shift_mm']:+.3f}|{x['integral_ratio']:.5f}|{x['mare_percent']:.3f}|\n"
text+="""
峰误差是GPU最大IDD / TOPAS最大IDD − 1；R80按各自峰高定义，记录GPU−TOPAS；MARE取TOPAS>10%峰值层。IDD均由3D横向求和。
各图文件以b1/b2/b3/b4开头，深度为TOPAS峰深度×1.2；原IDD相对误差±5%，另存不限y轴版本，避免掩盖超出范围的误差。
b1–b3包括sigma core/halo及误差，全部case包括9个深度的linear/log横向profile。

## 结果判断

本次RT07575完整wall为65.47秒，低于100秒；程序吞吐104.4k/s，比生产约3.03倍，比均值版下降约2.4%。这是单次配对配置测量，未测重复运行波动。
100/200/300 MeV/u的零能散峰高误差降至+1.83/+2.37/+0.83%，支持补回δ能损方差能显著改善峰区的判断；Gamma谱形近似与较长EM步的影响尚未分离。
该峰高指标不能代替逐层剂量误差或患者Gamma。200 MeV/u对数横向图仍显示GPU与TOPAS的低剂量halo偏差，生产版也存在类似趋势，不能据此声称所有横向尾部一致。
候选待用户审图，未接入默认生产，患者BODY Gamma尚未计算。

## 复现

build_moments_table.py → prepare.py → cmake（Release、SYCL nvptx64、sm_75）→ test_sampler.cpp → prepare_benchmarks.py / run_split_shard.py → benchmarks/run_gpu.py → benchmarks/compare.py → summarize.py。
候选源补丁和哈希见candidate.patch、source_manifest.json；原核数据保持v2.1并在CT运行前校验。
"""
(R/'README.md').write_text(text);print(text)
