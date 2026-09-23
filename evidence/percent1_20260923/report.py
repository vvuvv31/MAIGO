from pathlib import Path
import json,math,hashlib,tarfile,subprocess,shutil
r=Path('/mnt/sdb/wuwei/MAIGO_review_fix_20260923');o=r/'evidence/percent1_20260923'
j=json.loads((o/'comparison.json').read_text());assert j['status']=='COMPLETE'
def fmt(v):return f'{100*v:+.2f}%'
rows=[a for a in j['rows'] if a['window_width_mm']==.25]
lines=['| 深度 mm | 固定谷区差 | GPU 1SE / TOPAS贡献，百分点 | 合并95%区间 | 峰区差 | 平面总剂量差 |','|---:|---:|---:|---:|---:|---:|']
for a in rows:
 v=a['valley'];ci=v['CI95_approx'];lines.append(f"| {a['depth_mm']} | {fmt(v['relative_difference'])} | {100*v['gpu_only_SE']:.2f} / {100*v['topas_SE_contribution']:.2f} | [{fmt(ci[0])}, {fmt(ci[1])}] | {fmt(a['peak']['relative_difference'])} | {fmt(a['row_mean']['relative_difference'])} |")
passpoint=[a['depth_mm'] for a in rows if abs(a['valley']['relative_difference'])<.01]
passci=[a['depth_mm'] for a in rows if a['valley']['passes_pointwise_1pct_95CI']]
rejectzero=[a['depth_mm'] for a in rows if a['valley']['zero_bias_p_approx']<.05]
maxm=max(a['valley']['normal_95_halfwidth_1pct_history_multiplier'] for a in rows)
text=f'''# 1%目标：独立种子统计检验

**固定谷区六个深度，点估计在±1%内：{passpoint} mm；整个点位95%区间落在±1%内：{passci} mm。** 后者才支持该深度的统计验收，不能把单次点估计或“不显著”当成1%保证。这里更不声称所有体素均在1%内。

{chr(10).join(lines)}

全体积能量差 {fmt(j['energy']['relative_difference'])}。R80差 {j['r80']['gpu']['mean']-j['r80']['topas']['mean']:+.5f} mm；这是0.25 mm网格上的插值指标，不能理解为微米级几何准确度。

## 实验上下文

- 保留上轮修复：Cu最大步长0.05 mm，水Urban最大步长0.05 mm，Water_75eV，EM-only，C12 250 MeV/u，绝对每源粒子剂量，停止本领和散射幅度scale=1。
- GPU共15个独立种子，每个100万源历史：上轮3个 + 本轮新增12个。全部FP32剂量累计，各批单独运行后以高精度离线合并，避免扩大单次FP32累计量。
- TOPAS为原1000万历史参考，加本轮3个独立种子、每个200万历史，共1600万。每个新批128线程。单中心单spot；PBS多线程通用警告不能用来证明多spot分配正确，本实验没有跨线程多spot切换。
- 固定谷区 `abs(abs(x)-1.8)<0.45 mm`；峰区 `abs(x)<0.25 mm`。主指标仍为原0.25 mm深度层，20 mm等标称深度实际取19.875 mm等最近层。10 mm窗口只作为附加诊断。
- 固定12个新增GPU批次及3个TOPAS批次，未依据中途误差挑选种子或提前停止。

## 统计方法与粒子数

所有ROI先在单批内部求和/平均，再由独立批次估计方差，保留同一粒子对多个体素的相关性。没有把各体素误差当独立量相加。不同TOPAS批次按粒子数加权；对每源剂量 `x_i`，用 `c=sum(N_i*(x_i-mu)^2)/(k-1)` 估计单历史方差系数，合并均值方差为 `c/sum(N_i)`。比值采用delta近似和Welch-t 95%区间；TOPAS仅4批，区间本身仍有不确定性。FWHM及R80的批间误差是近似诊断，不替代线性剂量ROI验收。

以点位检验看，拒绝零差假设(p<0.05)的深度为 {rejectzero} mm。没有拒绝零差，不代表模型偏差已证明小于1%。六深度联合验收另用Bonferroni修正，见comparison.json。

若只要求最不利深度的**95%统计半宽约为1个百分点**，按本次方差和大样本1/sqrt(N)外推，两端同比增加约 {maxm:.2f} 倍，即GPU约 {15*maxm:.1f} M、TOPAS约 {16*maxm:.1f} M历史（均为总量估计）。这不保证偏差本身落入±1%；存在真实偏差时，更多粒子只会使偏差更清楚。若要证明|偏差|<1%，还需为实际偏差预留余量，并增加独立批次使方差估计稳定。

## 几何及过程核对

TOPAS 4.2.3 TsVGeometryComponent为Sc/XBins、YBins不同于实体分割数的评分器创建无材料平行评分副本；Geant4 PathFinder启用平行导航后，SafetyHelper取各几何的安全距离，ParallelWorldProcess也能限制步长。因此不能因其为平行评分网格就关闭GPU的体素截断/安全距离。细节上，平行网格边界与真实mass-world边界的stepStatus应区分；当前GenericIon fMinimal分支的tlimit重置不应自动推广为二者完全等价。此项未作为经验调参入口。

铜段能量涨落仍是近似高斯总能损，TOPAS离散电离过程不同；仅凭谷区剂量差无法确认它是主因。新入口相空间独立检查和电子迁移配对诊断另列对应JSON/图，均不用于对整体归一化或MCS幅度拟合。

## 证据与复现

- `comparison.json`：所有ROI、批次值、合并方差、区间、点位/六深度验收及粒子数外推。
- `gpu_quality.json`、`topas_quality.json`：运行质量。`run.py`为此轮运行脚本，`analyze.py`为统计及绘图脚本。
- `statistics.png`：主指标与附加10 mm窗口。
- 远端工作区：`{r}`；原始MAIGO_pristine未修改，无提交或推送。
'''
(o/'RESULTS.md').write_text(text)
print('REPORT',passpoint,passci,rejectzero,'history multiplier',maxm,flush=True)
