# Proton broadbeam FE 分段优化（2026-09-30）

## 结论

仅在专用 RT07575 质子 FE 配置中采用内部散射分段上限 **0.5→0.6 mm**。
外层几何/计分步长规则、平均能损上限2%、native能损涨落、核反应、带电次级、spot源和全部材料/package pins保持原配置；FE仍是core+tail。
二进制和核心源码不变，FP32 scorer ON / FP64 scorer OFF；全局及minibeam默认不变。

- 三个独立种子交替对照：391.940→411.376k/s，**+4.959%**。
- 八组完整分块交替对照：382.591→403.332k/s，**+5.421%**。
- 新完整计划：**250,630,000 histories / 623.826s = 401.763k/s**。primary 517.069s，secondary 61.807s。
- 旧设置此前完整计划377.799k/s。两次完整计划的温度/时钟与测量时间不同，不能把完整计划的直接比值全归因于此次分段修改；受控收益为上面的约5%。

## 剂量与审计

冻结TOPAS加权物理DIJ，源权重与照射条件不变。BODY∪GTV、参考≥10%最大剂量、local1%/1mm、DTA/10球形搜索、trilinear，无剂量拟合、坐标配准或再优化。

- 最终local11 **99.883094%**，当前已接受基线99.882159%，变化+0.000935个百分点。
- 20个分块全部quality accepted，队列溢出0，最大相对能量残差3.45e-07。
- 8个当前控制分块复现上一轮：每分块72084项整数审计一致，FP32剂量relativeL1约6.5e-8。
- FE分段修改会改变采样/后续轨迹与物理事件数，所以新候选不要求整数事件逐项等于旧设置；原quality、能量记账与队列门槛保留。

|指标|完整计划相对当前基线|20独立配对批次平均变化及t95半宽|
|---|---:|---:|
|D95_Gy|+0.01292%|+0.01318% ± 0.03489 pp|
|D98_Gy|-0.01742%|+0.02154% ± 0.07030 pp|
|D50_Gy|+0.00020%|-0.00715% ± 0.00850 pp|
|mean_Gy|-0.00065%|-0.00066% ± 0.00435 pp|
|entrance_line_mean|-0.03175%|-0.03179% ± 0.02601 pp|
|entrance_volume_mean|-0.00020%|-0.00020% ± 0.00110 pp|

这些固定关键指标的配对95%区间都在±1%内；8组相对最初保守GPU基线的区间也在±1%内（uncertainty.json）。分块DVH分位数的平均不是聚合DVH分位数；ROI平均/分位数检查不等于所有体素逐点1%认证。
既有quality research/non_production与物理近似标志保留。本病例为FE、local电子与neutral sink配置，不能推广为所有能量/患者或完整显式电子中子输运的速度与精度结论。

## 分段与筛选

独立100k诊断记录主粒子和带电次级所有FE调用的sum ceil(path/cap)：
0.5mm每call约1.654367段，0.6mm约1.000927段。
这是仪器计数估计值，不是逐次while循环日志，也不能单凭此断言所有省去的段都是“极短”段。诊断原子开销计时完全排除在正式速度之外。

0.505/0.51mm没有速度收益；1mm相对0.6mm的筛选增益仅0.65%，不足以支持更大的物理修改，均未选用，见failed.md。
提速主要在primary阶段；八对照中primary总时间219.316→205.865s，secondary 24.608→24.711s。

## 复现与输出

- `config/proton_rt07575_fe_fast.yaml`：接受后的单个12,531,500粒子分块配置。
- `run.py`：筛选/独立确认；`campaign.py`：原20个分块的整数spot分配和原种子。
- `finish.py`、`baseline_audit.py`、`uncertainty.py`：完整剂量、gamma与审计。
- `dose/full_cap060/dose.mhd`、`depth_dose.png/pdf`：按matRad原生轴与physical Gy归一化，1D仅BODY内。
- `diagnostics.json`、`confirmation_summary.json`、`completion_audit.json`、`full_runs.json`：原始指标。
- 原始GPU输出：`/mnt/sda/wuwei/proton_fe_cap_followup_20260930/runs/`。
- 当前二进制SHA256 `0d82beda3e03e4cd45ee05d73ccc321e055c7ac3b65119aec8da045725ba6629`，使用上一轮已验证的Stage7非离子EM专用核，无新编译。
- 全计划wall包括每分块程序包加载、初始化、输运与剂量写出；不包含外部合并/gamma/绘图及等待。

未commit/push。
