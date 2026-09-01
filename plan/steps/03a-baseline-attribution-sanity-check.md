# 03A：Compatibility baseline attribution sanity check

## 目标

在进入 03B-2B source/compiler cleanup 前，证明 `cinel02_topas_compatibility_mode`
本身没有改变 Be6 之外的物种 scorer 或上游 CINEL02 采样。该步骤只比较同一代码、
seed、package、rate、geometry 和 3D scorer 的 false/true A/B；不改变物理参数。

## 判定规则

- `Z4_Be` 与 `charged_total` 允许变化，因为 compatibility profile 会丢弃 Be6 的
  transport kinetic energy。
- Primary C、Secondary C、B、Li、He、Z1 的 GPU integral 必须在
  `0.01` percentage-point 内保持不变。
- 两份报告的 TOPAS reference integral、histories、energy、grid、scorer、package/rate
  hashes 必须一致。
- 该检查只能排除 scorer/analysis side effect，不能把 species integral 误差当作物理
  收敛结论；TOPAS reference/config 若不一致，A/B 直接判失败。

## 工具与证据

- 审计器：`startup/package_tools/check_cinel02_baseline_attribution.py`。
- 输入：`scripts/analyze_cinel02_species_idd.py` 生成的两个 `analysis.json`。
- false 配置：`config/beam_200MeVu_cinel02_e200light107_g1_transitiondiag4_100k_xy04.yaml`。
- true 配置：`config/beam_200MeVu_cinel02_e200light107_g1_topascompat_100k_xy04.yaml`。
- 输出：`plan/artifacts/cinel02-baseline-attribution-e200-g1/sanity.json`。

## 退出条件

- [x] 当前 HEAD 同 seed 的 false/true GPU A/B 已完成。
- [x] 稳定类别和 TOPAS/provenance 检查通过。
- [x] 旧 `becf880` 与新 compatibility baseline 的 TOPAS reference/config/scorer 差异已
  明确记录；未绑定 reference 的历史百分比只作历史对照。

## 完成证据（2026-09-01）

当前 HEAD `2791bae` 重建后，在本机 RTX 2080Ti/sm_75 以相同 200 MeV/u、G1、100k、seed
`2026095100` 完成 false/true A/B。checker 输出
`plan/artifacts/cinel02-baseline-attribution-e200-g1/sanity.json`，status=`pass`：
Primary C、Secondary C、B、Li、He、Z1 的最大变化为 `2.74e-8%`，均低于 `0.01`
percentage-point；Be aggregate `-11.5263%`、charged total `-0.0577%` 为预期允许项。
TOPAS reference scorer hashes、package/rate hashes、grid/scorer contract 全部一致。
旧 `becf880` manifest 未绑定 TOPAS reference config/hash，因此其百分比不用于解释当前
compatibility baseline 的整体抬升。

该步骤不修 sampler/yield/rate/stopping/MCS，也不以 dose 改善作为验收。若 sanity check
失败，先修 scorer/analysis 或 reference contract；通过后 03B-2B 仍只做 bounded
source/compiler correctness cleanup，不再期待其收敛 Be/Li 积分剂量。
