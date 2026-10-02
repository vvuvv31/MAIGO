# Proton / carbon CT broadbeam 调度优化评估（2026-10-01）

完成两类 CT broadbeam 的独立调度实验和反向顺序配对确认；本轮没有可采用的完整吞吐收益，正式配置、输运源码和 build/carbon_mc 均保持本轮前版本。上一轮 proton CT minibeam 的262144批次优化保留。本次使用本地 RTX2080Ti，未新开 Colab。

测试保留 FP32 dose ON、FP64 dose OFF、物理步长、电子响应、RNG 和全部审计。Proton 采用 RT07575 part_01 的12,531,500 histories；carbon 采用 RT07575 的6,481,909 histories。Proton 是一份计划分区，不是完整多分区计划。

| 算例/候选 | 原版完整秒 | 候选完整秒 | 完整吞吐变化 | 输运吞吐变化 | 两次完整吞吐变化 |
|---|---:|---:|---:|---:|---|
| Proton：primary 128→64线程 | 30.2133 | 30.1850 | +0.094% | -0.503% | +0.244%, -0.056% |
| Carbon：每4轮融合重分组 | 37.2687 | 37.2307 | +0.102% | +1.509% | +0.125%, +0.080% |

表中为两个独立种子1001800101/1001800102的平均；第一对候选→原版，第二对原版→候选。完整运行包含加载、校验、上传、输运和输出；输运循环包含计分和队列搬运。没有以单个kernel时间替代完整吞吐，也没有从微小差异推断稳定收益。

Proton 路线：

- 当前 non-ion Stage7 主核占主导，静态资源235 registers/thread、3680-byte stack；与早期约170-register的secondary工作组失败实验不同，因此建立新隔离候选。静态资源不是实际occupancy或spill等待占比的测量。
- 增加实验专用、严格取值的 CUDA BroadBeam primary local-size入口，筛选128→64→32→128。完整秒29.1615/30.2252/30.1572/30.2119，输运秒26.6712/27.1661/27.5384/27.4905，热机漂移明显；不以冷控制组和单次候选推断优劣。
- 再用两个独立种子与原始正式二进制比较64线程；完整差异约+0.094%，两对方向不一致，拒绝默认采用。128线程及既有69632-history配置保持不变。

Carbon 路线：

- 主要耗时在次级粒子。早期独立动态排序没有收益；本轮满足failed.md的重试条件：把当前(species,energy)分组融合到原有存活粒子压缩中。
- 每4或8轮以 histogram/prefix/scatter 替换原三个compaction kernel，另清零已有小计数器；复用ranks缓冲保存bucket，不搬动粒子状态、不增加独立排序pass。保持粒子原有state index、parent history和RNG身份。
- 0→4→8→0轮筛选完整秒37.2390/37.2964/37.2827/38.2758；随后独立确认每4轮方案。确认输运约+1.51%，但完整吞吐仅约+0.10%，初始化等成本抵消了小幅内核收益。没有证明值得采用的净收益，恢复/保留原稳定压缩路径。

质量、剂量与比较器：

- 所有完成运行的离散ledger/transport/EM计数相同，queue overflow=0，剂量有限且非负。
- Proton research quality全部接受。Carbon始终仅有原来的`unvalidated_electron_joint_response`失败、退出码1；其计时为diagnostic-only，未放宽质量门禁或声称生产验证通过。

| 算例 | 最大dose L1/参考总剂量 | 最大体素差/参考峰值 | 最大相对能量残差 |
|---|---:|---:|---:|
| proton_broadbeam | 6.36296e-08 | 4.99885e-07 | 3.1491e-07 |
| carbon_broadbeam | 5.79304e-08 | 9.58191e-07 | 3.55353e-06 |

- 修复`scripts/compare_minibeam.py`：`secondary_selected_energy_mismatch_sum`在C++中为double且来自SchneiderFloatSlot，即使JSON没有小数点也不属于离散计数。精确排除此字段的整数比较，并在本轮证据中保留其浮点变化；其余离散审计不变。Carbon第二对实际633986→633986.6875，相对1.0844e-6，严格1e-6标志为false但质量/总能量检查未失败；这是已允许的FP32累加顺序差异，不作为调度拒绝原因。
- 保留原始误判JSON/日志，重新分析原输出，仅补跑尚未开始的proton第二对，未重跑或筛掉carbon结果。

证据位于`out/broadbeam_schedule_20261001/`：`retry_basis.md`、`resources/`、`candidate.patch`、冻结`source/`及`build/`、`proton_blocks.json`、`carbon_regroup.json`、`confirmed.json`、`summary.json`，每次配置/原始剂量/能量账本/质量报告/telemetry完整保留。未采用路线和重试条件已追加至`failed.md`。本轮没有新的TOPAS gamma验证。

正式二进制仍为上一轮已验证版本：`2912e0761e028d735a0b3aa6f2c3960e75d86f05e1b0708b0d8850aae49089d8`。
