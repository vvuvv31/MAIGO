# RT07575 proton/carbon broadbeam 全计划电子包必要性验收

## 结论

以整个剂量网格、TOPAS参考≥10%全体积峰值、local1%/1mm >99%（球形搜索0.1mm）验收：本轮质子关闭电子包仍通过，碳离子关闭后不通过，应保留碳宽束电子包。如果验收范围限定在患者BODY内，两种粒子关闭后都通过。此前碳的99.874%只针对BODY，不能代替全网格结果。

| 粒子与评估域 | 体素数 | 开启当前电子包 | 关闭电子包 | 关闭是否>99% |
|---|---:|---:|---:|---|
| Proton，BODY∪GTV且≥10%peak | 320771 | 99.951991% | 99.948873% | 是 |
| Proton，全网格且≥10%peak | 416837 | 99.909317% | 99.339310% | 是 |
| Carbon，BODY且≥10%peak | 351014 | 99.874934% | 99.873794% | 是 |
| Carbon，全网格且≥10%peak | 460431 | 99.901614% | 96.225276% | **否** |

阈值固定在各自TOPAS参考；两个GPU模式使用完全相同的评价体素，搜索使用完整GPU剂量体积，禁止外推。没有拟合剂量比例、位移或旋转。质子、碳是各自原有计划，不能把不同计划的绝对Gy直接互比。患者网格0.5×0.5×2mm，未重采样为0.5mm各向同性。

## 全网格与体内为什么结论不同

BODY外的≥10%peak区域，质子开启/关闭通过率99.766827%/97.303937%，碳99.987205%/84.520687%。关闭后碳全网格17380个失败体素中，16937个在BODY外，16929个同时满足ρ<0.3g/cm³，占全图失败的97.405%；质子2754个失败体素中，2582个在BODY外低密度区域，占93.755%。主要差别位于体外低密度介质，不能把该结果概括为所有区域电子剂量都不重要。

关闭−开启的变化如下；RMS为逐体素相对开启剂量的差，dose sum为区域剂量值求和变化，不是总沉积能量。

| 粒子/区域 | 区域dose sum变化 | RMS局部剂量变化 |
|---|---:|---:|
| Proton，体内评价区域 | -0.005294% | 0.200775% |
| Proton，全网格评价区域 | +0.322908% | 2.418576% |
| Proton，BODY外评价区域 | +1.827333% | 5.024622% |
| Carbon，体内评价区域 | -0.013530% | 0.333749% |
| Carbon，全网格评价区域 | +0.652273% | 3.237000% |
| Carbon，BODY外评价区域 | +3.347248% | 6.613259% |

关闭模式将原本独立电子/电磁光子的空间输运替换为现有离子路径局部计分。这种改变在空气中更明显；宽束体内评价可被相邻束斑补偿以及1mm距离搜索容忍。本轮只定位了误差范围，没有把全部体外差异逐事件归因。γ通过不等于逐体素误差<1%，也不证明电子空间重分布物理可以普遍删除。

## 本轮源、统计及参考

每个粒子每个GPU模式16×1M独立histories，连续复用一个GPU context/packages，按整个16M campaign一次最大余数分配权重，再分批；配对模式的每spot源和seed相同。不是只照一个spot或局部场。

质子先完成默认预设的两模式16M对照，保留在`out/proton_ct_no_electrons_20261009/high_statistics/`，但该默认源是原计划part_01整数分配的相对权重。本报告的质子正式验收**另跑**原始完整优化计划：870行、540个正权重spot，使用`/mnt/sdb/wuwei/MAIGO_review_fix_20260923/benchmark/protonCT/RT07575_proton_GPU/inputs/spots_optimized.csv`，并逐字节固定它。原完整源未被part_01反复分配替代，CT和beam model与该历史参考逐字节一致。正式输出位于`out/proton_ct_no_electrons_20261009/full_plan/high_statistics/`。

质子TOPAS参考为该目录冻结的`inputs/physical_dose.raw`：原TOPAS物理剂量DIJ按原优化权重重建，非生物剂量，非新的TOPAS模拟。参考、patient.mat、CT、beam model和source哈希均核对原manifest；参考数组同时与此前已审计的重建逐点一致。每臂批次实际Gy求和后，只乘一次25062649959.598873/16000000，换算至原优化单fraction。BODY∪GTV评价掩模沿用已经核对MATLAB ROI轴映射的历史掩模。

碳复用上一轮独立16M/臂的完整计划输出与已经核对的历史TOPAS宽束参考，853个正权重spot。参考为5个完整plan replica共129638170粒子的DoseToMedium Sum；每臂批次真实Gy求和后，只乘一次129638170/16000000。原TOPAS源的零权重行被跳过，GPU输入历史整数分配与TOPAS归一化权重L1差0.003190%，对两个GPU模式相同。详见`carbon_ct_no_electrons_20261009.md`。

两个计划都使用既有verified packed_xneg映射：`flip(gpu_zyx.transpose(1,2,0),axis=2)`。本报告用CPU gamma evaluator，local1%/1mm、search0.1mm，与既有判据一致。

## 物理范围与审计

开启模式是当前broadbeam选择性电子/电磁光子输运，density threshold0.1g/cm³、halo5mm、group32及survival0.5。不是将完整显式电子模型与关闭模式相比较。关闭实验只使用现有`broadbeam_electron_tracking:false`路径并删除四个电子/光子包入口；没有缺包后静默局部沉积，没有改continuous26离子EM、涨落、0.5mm步长、核反应或带电碎片。核中性产物仍为未追踪能量汇。

64个正式GPU批次全部quality accepted，energy/EM/coverage/completion和zero-overflow检查通过。每个粒子的16组配对source CSV逐字节相同；核事件、弹性事件、live ion terminal计数及离子transport ledger匹配。quality的research/non_production标记保留；运行审计不等于物理验收。

独立GPU批次提供单体素SE估计，报告JSON保留数值；历史TOPAS DIJ的逐体素方差不可用。上述通过率是本轮统计量下的观测值，不声称对所有患者或无限统计量的物理等价。没有修改八个发布预设、物理包或运行时代码。

## 性能与复现

前述正式ABBA观测：质子开启152659/s、关闭298837/s，约1.96倍；碳开启75509/s、关闭105424/s，约1.40倍。质子ABBA使用默认整数分块权重。本轮原完整源的质子顺序剂量campaign暖批吞吐151853/s与288198/s；顺序campaign只作运行记录，不用其直接比值替代交错性能归因。

总报告、对比原始dose、gamma失败掩模和分析脚本：`out/rt07575_broadbeam_electron_gate_20261009/`。`report.json`记录两个域及体外/低密度区域、归一化和审计，`analyze.py`可复现；需先完成质子full_plan campaign。一次提前分析被完整批次数guard阻止，日志保留为`analysis_incomplete_guard.log`，不作为正式结果。

因此，对本次既定验收：**质子可作为关闭电子包的候选；碳若要求全网格>99%，必须保留当前电子包或另行验证替代方案。** 若今后明确只验收患者内剂量，两者本轮都通过，但这不是当前全网格门槛的替代。
