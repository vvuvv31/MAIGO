# Unrestricted 短步均值修复（2026-10-20）

## 改动范围

`UnifiedEmPoint::mean` 现在按 EMCNT001 的 `is_ion` 模型标记处理水/CT 短步。
非 ion 模型（包括质子，不是中性粒子）在步首一阶能损未超过 `linear_limit * E`
时使用 `S_full(E_pre) * true_path`；ion 模型保留原有中点求值。
超过阈值仍使用 unrestricted 剩余射程反演；到达射程端点仍返回剩余动能。

这与原生 `G4VEnergyLossProcess::AlongStepDoIt` 的非 ion 短步分支一致。
原生 ion 另有 `CorrectionsAlongStep`，本补丁没有改动现有 ion 近似。
未改变 unrestricted stopping、range/inverse、涨落、Cu/air、核反应、步长、源或剂量归一化。
按物理模型标记选择，不按病例或密度拟合；carbon 计划中的非 ion 碎片也会经过此规则。

## 修复依据：已有同统计量因果试验

20022516 proton broadbeam，GPU/TOPAS 各 361,050,622 粒子。
BODY 内 TOPAS >10% Dmax，local gamma，搜索步长 0.1 mm；不缩放或配准剂量。

| 实现 | local 1%/1 mm | local 3%/0 mm |
|---|---:|---:|
| unrestricted 迁移前 | 99.45092% | 94.72830% |
| unrestricted，全部短步中点 | 99.01459% | 92.80532% |
| unrestricted，非 ion 步首修复 | 99.44825% | 94.61902% |

在原本净增的 3,274 个失败点中净恢复 3,254 个（99.39%）。
低密度 rho<0.3 g/cm³ 区域平均相对偏差从 −0.32283% 恢复至 −0.06557%；
迁移前为 −0.06016%。37 批次 CSV、spot 分配、种子与 TOPAS 参考完全相同。
所有运行审计通过，零溢出，最大相对能量残差 2.6161e−7。

这些是已完成候选的结果，本次没有重复该 3.61 亿粒子试验。
本次修复头文件与该候选除注释/空白外 token 完全相同；包含其他当前改动的源码另行重建检查。
原始证据：`tmp/lung_unrestricted_repair_20261019/full_native/RESULTS.json`。

## 本次当前源码检查

检查在 `/mnt/sdb/wuwei/MAIGO_unrestricted_lung_analysis` 隔离工作树完成。
GPU 为 RTX 2080 Ti，所有五类管线 AOT `sm_75`，`CUDA_DISABLE_PTX_JIT=1`。
A6000 被其他训练任务占用，本次补充检查没有占用它；上述高统计量试验曾使用两卡。

离线均值规则检查通过：18,109 个非 ion 短步、32,851 个反射程分支、
47,320 个保留的 ion 中点分支及 32,760 个端点检查。

同批次配对结果如下。宽束取原全计划前四批，微束取前两批，各约 4,000 万粒子。
参考 TOPAS 和 BODY mask 固定，宽束使用 TOPAS >10% Dmax、微束使用 >0.4 Gy。
本轮只用于配对回归，不能代替十亿级微束最终验收，也不作为吞吐测试。

| 配置 | 粒子数 | 指标 | 修复前 | 修复后 | 3%/0 mm：前 → 后 |
|---|---:|---|---:|---:|---:|
| proton_RT07575 | 38,557,936 | local 1%/1mm | 99.99191% | 99.98911% | 94.81390% → 95.24128% |
| proton_RT06423 | 38,235,807 | local 1%/1mm | 99.99642% | 99.99438% | 96.69986% → 96.95187% |
| proton_minibeam_RT07575 | 39,999,985 | local 3%/0.2mm | 93.44276% | 93.44276% | 61.36842% → 61.43565% |
| proton_minibeam_RT06423 | 40,000,011 | local 3%/0.2mm | 96.21187% | 96.22079% | 70.38344% → 70.72239% |
| proton_minibeam_20022516 | 40,000,014 | local 3%/0.2mm | 79.80868% | 80.02512% | 45.75492% → 45.96348% |

五组逐 spot 计数及种子不变；共完成 196,793,753 粒子，
全部能量/EM/覆盖/完成度审计通过且零溢出。八个发布预设另各运行 2×20,000 粒子，
均通过运行审计；这些短检查不证明碳剂量精度，也不要求物理改动前后事件计数相同。

Lung 微束该样本的区域平均相对偏差从 +0.070% 变为 +0.588%，虽然 gamma 提高，
不能宣称所有误差指标都改善，亦不能据此认定微束原有差异已经解决。
碳主离子均值分支未改变；碳计划里的非 ion 碎片会受影响，本次仅完成其预设运行检查。

详细补充证据保留在隔离树 `tmp/mean_rule_repair/`：
`MEAN_LAW.json`、`HIGH_STAT_CANDIDATE_MATCH.json`、`CROSS_PLAN.json`、
`RELEASE_RUNTIME.json`、逐批剂量和质量报告。测试驱动不合入发布源码。
