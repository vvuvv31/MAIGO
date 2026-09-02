# 04D：TOPAS isotope lineage/survival reference

## 目标

为 6Li、7Li、7Be、9Be、10Be 建立与 GPU episode ledger 同语义的 TOPAS/Geant4
参考：按 hadronic-cascade lineage 推断 generation，并对每个 transport episode 记录
birth energy、路径长度、首次后续 hadronic interaction、interaction energy 和终止原因。
本步骤只增加 TOPAS diagnostic scorer，不改变 TOPAS physics 或 GPU runtime。

## 实现

- 新增 CarbonLineageSurvivalNtuple。
- 仅追踪 Li-6/Li-7/Be-7/Be-9/Be-10 的 hadronic products；primary Li/Be 不进入报告。
- primary generation 记为 -1，因此 primary 的 direct hadronic products 为 G0，后续
  hadronic descendants 依次为 G1、G2。
- 每条 track 按 episode 累计 step path；首次后续 inelastic interaction 输出
  reacted=1，若父 track继续则从 post-step state 开启下一个 episode。
- stop、decay、几何边界和 event-end 未结束轨迹分别记录 terminal reason；event-end
  记录为 right-censored。
- 新增 summarize_topas_lineage_survival.py，严格校验 header/data 行数、列、episode identity、
  reacted/reason/censored partition，并输出 isotope×generation CSV/JSON。
- 新增 compare_topas_gpu_lineage_survival.py，保持 TOPAS generation 与 GPU 当前 G1
  policy total 的语义差异，不把 descriptive smoke 当作 acceptance gate。

## Canonical TOPAS smoke

- 本机 TOPAS 4.2.3 / Geant4 11.3.2，CPU sbatch job 497。
- 200 MeV/u、100,000 histories、seed 2026095100、H1/O16、3D/phantom component。
- 初次 job 496 因既有 exposure scorer 的 501 MeV/u diagnostic grid 越界中止；脚本随后
  将完整 C-12 动能预算覆盖扩大到 5001 个 1-MeV/u bin，并用新 campaign UUID 重跑成功。
- 输出：/mnt/sda/wuwei/cinel02-lineage-survival/e200MeVu_100000h_lineage_survival_v2/。
- 汇总：plan/artifacts/topas-lineage-survival-e200-g1/summary.json；
  对照：同目录 comparison.json。

## 结果

TOPAS 100k 输出 7,717 条 episode，0 条 event-end censored，终止原因只有 stopped
和 hadronic_interaction。按 generation 的主要统计如下：

| isotope | G0 episodes/reacted | G1 episodes/reacted | G2 episodes/reacted |
|---|---:|---:|---:|
| Li6 | 2137 / 258 (12.07%) | 1049 / 12 (1.14%) | 155 / 0 |
| Li7 | 1282 / 267 (20.83%) | 568 / 9 (1.58%) | 95 / 0 |
| Be7 | 1161 / 216 (18.60%) | 249 / 3 (1.20%) | 33 / 0 |
| Be9 | 296 / 47 (15.88%) | 297 / 4 (1.35%) | 88 / 0 |
| Be10 | 151 / 36 (23.84%) | 106 / 3 (2.83%) | 15 / 0 |

GPU 当前 G1-policy total（queued birth → reaction-killed）为：

| isotope | episodes | reacted | reaction fraction | tau runtime |
|---|---:|---:|---:|---:|
| Li6 | 3040 | 278 | 9.14% | 267.18 |
| Li7 | 1703 | 263 | 15.44% | 279.27 |
| Be7 | 1437 | 203 | 14.13% | 229.55 |
| Be9 | 478 | 31 | 6.49% | 36.27 |
| Be10 | 231 | 49 | 21.21% | 40.11 |

这些数值是同一 100k smoke 的描述性结果；TOPAS 与 GPU 的 product sampling、
物理实现和 generation policy 仍不同，不能直接用 episode count 的差异判定
package/yield bug。它们提供了下一步 reaction survival / stopping residence 因果分析
所需的 reference occupancy 和终止语义。

## 验证

- TOPAS extension Release build 成功，binary 含 CarbonLineageSurvivalNtuple。
- 本机 TOPAS job 497 成功完成，real time 约 64 s。
- lineage summarizer 成功解析 7,717 行并生成 isotope×generation 汇总。
- synthetic test test_isotope_generation_partition_and_header_units 通过。
- GPU 仍使用既有 04B/04C ledger；本步骤未改 rate、target selection、generation gate、
  stopping、straggling 或 MCS。

## 退出结论

- [x] TOPAS lineage/survival episode reference 已建立。
- [x] generation、path、birth/interaction energy、terminal reason 可按 isotope×generation
  审计。
- [x] TOPAS/GPU 对照报告已生成。
- [x] 未把 descriptive 100k 结果当作 physics acceptance gate。

## 遗留问题

- TOPAS scorer 目前按 component 记录 path；尚未输出 target identity 和连续 optical depth，
  因此下一步仍需将 reference outcome 与 GPU H/O hazard 分开比较。
- TOPAS 与 GPU 使用不同 event generator/transport semantics，episode 数和 reaction
  fraction 不能直接作为 yield 验收。
- 尚未做更高统计或多 seed survival reference；下一步基于该 smoke 进行 TOPAS/GPU survival 对照与 Be/Li 因果审计。
