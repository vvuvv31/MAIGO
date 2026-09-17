# 文档导航

更新日期：2026-09-05。当前说明与历史记录分开维护；研究验证不等于临床准入。

| 文档 | 唯一职责 |
|---|---|
| [仓库入口](../README.md) | 构建、运行前检查、入口导航 |
| [AGENTS](../AGENTS.md) | 执行约束和最低数据版本；历史文档不得覆盖 |
| [structure](structure.md) | 当前源码与构建结构 |
| [Physics model](TOPAS_GPU_Physics_Model.md) | 当前 CT / water 路由、近似与验证边界 |
| [Planning / geometry](planning.md) | spots、history 分配、坐标与 CT 重排 |
| [Scoring / validation](scoring_validation.md) | DoseToMedium、LET、ledger、Gamma 口径 |
| [当前结果索引](results.md) | 冻结运行、指标和证据入口 |
| [Materials and Methods](../mm.md) / [中文](../mm_zh.md) | 与实现说明同步的双语方法稿，不另定参数 |
| [活动数据](../data/ACTIVE_DATA.md) | 保留数据、外部依赖、trash 恢复方式 |
| [Schneider 计划](../plan/README.md) | 原 workstream 进度与验收门禁 |
| [电子响应计划](../plan2/README.md) | 新响应候选进度；以当前状态表为准 |
| [FRED 论文解读](FRED_Carbon_Fragmentation_Model.md) | 文献参考，不是当前实现规格 |
| [历史归档](archive/README.md) | 旧设计、实验、方法稿和会话证据 |

修改规则：实现以源码为准，数据版本以 bundle/manifest 和 AGENTS 为准，结果以冻结的
executable、输入哈希、运行配置和证据为准。发现不一致先记录并核实，不由旧文档推定
功能已实现或门禁已通过。不从归档恢复旧物理参数，不自动提交或推送。
