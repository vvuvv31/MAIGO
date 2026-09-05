# 历史归档

本目录不是当前执行说明。旧参数、旧包、自动 commit/push 指令、旧门禁和旧
“DONE”结论不覆盖 [AGENTS](../../AGENTS.md)、[现行文档](../README.md) 或当前计划。
归档不等于已证伪：理论推导、反向工程、失败实验和唯一原始记录仍应保留。

## 2026-09-05 文档整理

18 份原文件按原始路径分为 `root/` 与 `docs/`，移入
`2026-09-05/`。原文件逐字节保留，SHA256 见
[relocation_manifest.json](2026-09-05/relocation_manifest.json)。
未永久删除内容；重写的 README、structure、physics、mm 有完整旧版备份。
历史相对链接保持原样，可能不再从归档位置直接可点击；请用 manifest 的
`original` 解释其原始基准目录，不为修链接篡改冻结原文。

| 原位置 → 归档 | 性质 / 当前去向 |
|---|---|
| [README.md](2026-09-05/root/README.md) | 旧入口/构建说明；当前根 README 已重写 |
| [A_Data-Driven_Fragmentation_Model_for_Carbon_Therapy.md](2026-09-05/root/A_Data-Driven_Fragmentation_Model_for_Carbon_Therapy.md) | 重复论文解读；现行唯一副本为 docs/FRED_Carbon_Fragmentation_Model.md |
| [model.md](2026-09-05/root/model.md) | FRED 逆向证据与旧 MAIGO 映射，保留独有证据，不作当前规格 |
| [mm.md](2026-09-05/root/mm.md) | 旧英文方法稿（含独有多离子/质子章节） |
| [mm_zh.md](2026-09-05/root/mm_zh.md) | 旧中文方法稿，不假定与英文完全一致 |
| [scorer_benchmark.md](2026-09-05/root/scorer_benchmark.md) | 旧 scorer 建议稿；已整理为 scoring_validation |
| [issue.md](2026-09-05/root/issue.md) | 旧 water 问题列表；旧 open/closed 不转移为当前状态 |
| [plan.md](2026-09-05/root/plan.md) | 旧 CINEL02 修复建议；不是 plan/README 进度控制 |
| [session.md](2026-09-05/root/session.md) | 历史会话与工具转录，含被后续撤销的推断 |
| [docs/BRANCH_WORKFLOW.md](2026-09-05/docs/BRANCH_WORKFLOW.md) | 旧 master/自动 push 指令失效；不得执行 |
| [docs/PATH_A_FRED.md](2026-09-05/docs/PATH_A_FRED.md) | 历史 FRED/water 路线 |
| [docs/PROJECT_PROGRESS_2026-08-22.md](2026-09-05/docs/PROJECT_PROGRESS_2026-08-22.md) | 历史多离子/电子/minibeam 进展 |
| [docs/TOPAS_GPU_Physics_Model.md](2026-09-05/docs/TOPAS_GPU_Physics_Model.md) | 旧物理说明，保留旧版本比较 |
| [docs/structure.md](2026-09-05/docs/structure.md) | 旧架构图与文件清单 |
| [docs/geometry_rotation.md](2026-09-05/docs/geometry_rotation.md) | 几何旋转警示原稿；现行内容合并到 planning |
| [docs/ctplan.md](2026-09-05/docs/ctplan.md) | 旧 geometry/DIJ/LS-scale 工作流及历史结果；有效约定合入 planning |
| [docs/ctResult.md](2026-09-05/docs/ctResult.md) | 历史 CT/LET 结果；当前结果另建索引，不改旧数值 |
| [docs/minibeamStructure.md](2026-09-05/docs/minibeamStructure.md) | 旧 minibeam 设计，不据此认定当前 kernel 存在 |

现行入口：
[structure](../structure.md)、[physics](../TOPAS_GPU_Physics_Model.md)、
[planning](../planning.md)、[scoring](../scoring_validation.md)、
[results](../results.md)、[English methods](../../mm.md) / [中文](../../mm_zh.md)。

## 此前已有归档

- [futureStep](futureStep.md)：旧路线建议，不直接恢复为待办。
- [local30](local30.md)：历史严格 Gamma/几何调试；旧调参建议不再作为指令。
- [minibeam](minibeam.md)：历史设计。
- [CT match checklist](MAIGO_TOPAS_GPU_Benchmark_CT_Match_Checklist.md)：
  provenance、统计和 overflow 原则已体现在现行 scoring；旧案例结论保留。
- [Geant4 straggling](geant4_energy_straggling.md)：理论参考，
  不因为年代较早删除；用于新实现前需核对 Geant4 版本与实际算法。

## 恢复与数据档案的区别

文档归档在 docs/archive，计划纳入 Git；不要移动到 ignored trash。
物理数据归档另在 `trash/data_archive_20260905/`，恢复方式见
[ACTIVE_DATA](../../data/ACTIVE_DATA.md)，新 clone 不自动含这些数据。

需要查旧版本时优先阅读归档，不覆盖现行文件。确需恢复，应先核对 SHA，
另存新位置并标明历史身份；当前文件和现有用户修改不得被覆盖。
