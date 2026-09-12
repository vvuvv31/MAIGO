# 全离子 elastic 验证记录（2026-09-11）

覆盖既有 18 种带电 projectile、13 个靶元素、25 个 Schneider 材料及水。额外天然重反冲采用 37 同位素 stopping 表，只做减速/MCS，不扩展其核反应。

`summary.json` 保存最终可执行文件 SHA、测试数量与质量报告；`source_pins.json` 保存实现源码指纹。最终水 50k 和 RT07575 6,481,909 histories 分片均通过研究运行闭合检查，零溢出。`*_initial.json` 是早期构建结果，不作为最终构建证明。

`shard_comparison.json` 由 `compare_shard.py` 对最终分片计算：BODY 内参考剂量 ≥10%，搜索步长 0.125 mm，仅按 histories 缩放 TOPAS 参考，无剂量拟合。旧无弹性对照也是单分片。历史 TOPAS 可能未启用同一 CarbonIonElasticPhysics，因此此 Gamma 仅为变化诊断，不是匹配物理验收，也不能与全统计结果直接比较。

TOPAS 实际弹性模型：proton hElasticCHIPS；d/t/He3/alpha hElasticLHEP；其余 NNDiffuseElastic。数据二进制、提取来源及源码快照由 `tools/verify_all_ion_elastic.py --audit-provenance` 审计。旧 Schneider v2.1 栈不变。

尚需相同 TOPAS 物理列表、样本/能量节点收敛及全病例全统计验证；生产模式仍禁止该候选包。11.3.2 Release 不含新包。
