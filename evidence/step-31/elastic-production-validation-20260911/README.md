# 默认生产 elastic 接入验收（进行中）

历史运行日志审计见 `historical_process_audit.json`。已确认审计到的 GenericIon 进程块不含独立 elastic；历史参考不能直接作为新全离子弹性的匹配物理验收。

本机工作目录：`/mnt/sda/wuwei/elastic_production_validation_20260911`。全部 TOPAS 作业显式限定 `--nodelist=ps`，当前主机和 compute 唯一节点均为 ps。保留旧参考及旧物理包。

- `rt07575_matched_pilot`：同病例几何、源和 3D dose，保留原7模块并增加 CarbonIonElasticPhysics；约52k histories，试测时间和内存。
- `matched_process_audit`：开启物理 verbosity 的小样本进程审计。首试因各spot取整为零失败，调整scale后重新提交；不作为剂量验证。
- `samples2048`：18物种、原网格、独立seed、每节点2048末态，冻结原提取器；由 `tools/compare_elastic_sample_banks.py` 与512库比较。
- `grid40`：18物种、每十倍能量40节点、512末态，独立编译提取器；用于能量网格收敛。
- `gpu_samples2048`：冻结GPU二进制和RT07575原始完整分片，仅替换2048样本候选包。

第一轮 18,598 个有效通道的 t/tmax 两样本 KS 比较，在 family-wise alpha=0.01 的 Bonferroni 校正下没有显著通道。此结果不等于稀有尾部、剂量或能量节点收敛通过。摘要及完整逐通道报告的路径/SHA在 `sample_convergence.json`。

## 后续通过条件

1. 匹配参考的18物种进程、截面、模型以及几何/源/计分/归一化有可审计来源。
2. 候选包通过来源、schema、全通道验证与host/device查询；水50k闭合、CT单分片对照零溢出。
3. 同统计量、相同源下比较样本和能量网格变化；报告剂量差、射程、尾部及Gamma，不用“KS不显著”替代剂量收敛。
4. 量化额外EM-only反冲的能量和空间贡献，检查停止本领中的核stopping与显式elastic的适用关系。
5. 三病例全统计、BODY内参考剂量>=10%，固定归一化、0.125mm搜索，输出33/22/11/30 global/local；检查局部失败区域和吞吐量。正式接受前记录数值阈值与结果，不能只凭单分片Gamma改善。
6. 全部通过后才冻结manifest、发布新数据并解除production候选限制。本文件不构成生产验收；尚未切换默认配置或发布。

## 已完成及正在运行

直接进程审计：旧列表只有5/18带电物种具有elastic，新列表18/18均恰有一个elastic；实际进程与模型逐物种匹配当前bank的提取记录。参见 `*_projectile_processes.json`。

2048样本及40节点/decade两套候选包，各通过2808 host/device查询与6481909 histories CT分片，零溢出。Gamma变化见 `dose_convergence_gamma.json`：历史TOPAS参考仅用于诊断，不能作为匹配物理接受。发现低能截面起始区插值误差，见 `grid_rate_convergence.json`，尚未宣布网格收敛。

约51852 histories的匹配TOPAS试跑完成，8线程耗时1064.77秒，峰值RSS约5.1GiB。三病例全统计匹配参考已经提交：RT07575=4613，RT06423=4614，20022516=4615；每个数组最多一个副本运行，各64应用线程/48GiB，总上限192线程/144GiB，节点固定为本机ps。配置及来源固定在 `matched_reference_campaign.json`。预计需要数天，依赖可用调度资源；提交不等于完成。

新提取器增加 EnergyNodesPerDecade 和 AuditAllProjectiles，仅用于明确的收敛与进程审计；默认能量网格仍为20节点/decade。旧recoil来源源码保存为独立快照，原SHA不变，全来源验证仍通过。

正式参考运行中出现 TOPAS Patient_Y_Division 未计分步骤提示（初始日志单步能量约1.46e-10 MeV）。必须在每个副本结束后汇总 unscored steps/energy、源histories与输出文件，确认计分损失可接受，不能仅凭作业退出码宣布参考有效。
