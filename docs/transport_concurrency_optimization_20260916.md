# 输运内核并发与访存等待复测（2026-09-16）

本轮以开始实施时的脏工作树快照为基线。结论是：没有新候选达到 RT07575 端到端
吞吐提高 5% 的推广门槛，因此默认仍为 64 路审计分片、16 步次级续跑；所有研究候选
保持关闭，没有修改生产配置，也没有 commit/push。

## 冻结基线与协议

冻结目录为 `scratch/transport_concurrency_20260916/`。`snapshot.json` 记录 Git HEAD、
工作树状态、源码/未跟踪构建依赖、数据和外部输入的 SHA256；`source/` 是独立源码与
数据副本，CT、spots 和 beam model 复制后重写到冻结配置。基线二进制 SHA256 为
`1202cd92c2ae54f129a53e6207ade7018729ecb130848a70df7641575189dfca`，冻结配置为
`f5897f9ee700ad2da53a57a43ca1fa97300d57c4446c75fd16b999d8c4e11579`。

运行固定 RTX 2080 Ti/sm_75、Schneider v2.1、统一 EM、两矩 delta 包、原生涨落
`scale=1`、精确 CT 面、16 步续跑、8192 尾部阈值和精确索引。Schneider manifest、
统一 EM `8c5d...855` 和 delta moments `c551...7cc` 均在每组计时前严格校验通过。
计时与 profiler 分开；计时脚本记录进程 wall、程序 Elapsed、原发/次级 kernel、GPU
温度/时钟和3D dose，并在 quality、能量记账或 overflow 失败时立即停止。

基线先预热2次，再正式运行5次：wall 中位 24.7112 s（24.2375–25.0588），Elapsed
中位 22.9900 s（22.5770–23.3149），原发 8.4345 s，次级 10.7545 s。全部 quality
通过、零 overflow、8项统一 EM 审计整数一致。5次3D剂量的全局最大波动包络为峰值的
0.00003908%。温度范围61–84 °C，SM时钟1350–1860 MHz。

## 硬件归因

Nsight Compute 对每个阶段做3次独立应用运行，保持 `clock-control=none`。阶段由
`generation begin + continuation round + active count` 选择，而不是按不稳定的内核序号
冒充匹配。实际入口和中位结果如下：

|阶段|活动粒子|block|寄存器/线程|寄存器驻留上限（block/SM）|active warp/scheduler|eligible warp/scheduler|long-scoreboard cycles/issue|
|---|---:|---:|---:|---:|---:|---:|---:|
|原发稳态|34,816/批|128|172|2|1.976|0.0970|9.31|
|次级早段 gen1/r0|9,337,479|32|186|8|1.950|0.0862|10.29|
|次级中段 gen1/r17|1,760,852|32|186|8|1.994|0.0853|14.66|
|次级晚段 gen1/r34|10,533|64|186|4|1.296|0.0678|11.39|
|第二代 r0|3,855,348|256|186|1|1.943|0.0881|15.03|

原发 L1/L2 命中率中位65.61%/85.90%；次级随阶段变化，完整数值在
`benchmark/transport_concurrency_20260916/hardware_profile_summary.json`。SASS/source
报告中存在大量真实 `LDL/STL`，并且 NCU
报告了 local-memory sectors；这证明线程私有 local memory 有实际流量。报告不把栈帧
大小直接等同于 spill，也不把 long-scoreboard 比例直接解释成某模块耗时。

## 单项候选

所有候选均由冻结源码独立构建，保持物理配置和随机种子。
`benchmark/transport_concurrency_20260916/candidate_screen_summary.json`
是机器可读的探索性淘汰结果；探索性单配对只能否决候选，不能用于推广。

|候选|Elapsed吞吐变化|原发变化|次级变化|决定|
|---|---:|---:|---:|---|
|审计1路|−2.21%|+0.72%|−5.13%|淘汰：原子竞争回归|
|审计32路|+0.10%|−0.45%|−0.16%|淘汰：无5%收益信号|
|审计128路|+0.06%|−0.34%|+0.38%|淘汰：无5%收益信号|
|审计256路|−2.48%|+0.02%|−0.13%|淘汰：端到端回退|
|共享精确搜索|正式配对+0.59%|+0.74%|−0.01%|淘汰：5次配对95%区间不含5%|
|细索引|−5.19%|−5.43%|−5.94%|淘汰：额外索引访存回退|
|原发初始化kernel|−0.48%|−0.65%|−0.48%|淘汰：额外状态/launch回退|

共享搜索正式5次配对的 wall/Elapsed 平均改善分别为0.57%/0.59%，95%配对 bootstrap
区间分别为−1.06%–2.20%和−0.90%–2.09%；5%不在区间内，所以无需扩展到10次。
全部正式运行 quality 通过、零 overflow、8项审计一致。候选最大3D体素差为峰值的
0.00004263%，低于该组基线0.00004618%的全局波动包络。真实包查表门槛覆盖3150条
材料/离子记录和25,872,175个节点，检查节点、区间
边界及相邻浮点值；共享搜索和细索引与原路径逐位一致。另用201,600组真实材料密度、
18种离子、随机输入检查涨落结果及 RNG 调用次数/终态，失败数为0。

已有同一工作树的5次配对还确认256与64路约−0.11%、1024与64路约+0.04%，64路已在
平台区。线程块只应在寄存器需求改变后重扫；本轮获胜组合不存在，初始化分离也已经
端到端回退，因此没有用强制寄存器上限或再次推广32/64/128固定块。

## 分阶段 EM 候选的停止条件

既有真实输入微基准覆盖1,405,166次受限涨落采样：独立 sampler kernel 约1.20 ms，
而本轮次级 kernel 每个分片约10.75 s。即使把 sampler 本身变成零成本，也远不足5%。
完整请求方案还必须每个物理步写入/读回粒子身份、确定步长、采样输入、RNG计数器和
结果，并增加请求、采样、继续三段 launch；16步续跑下会把每段变成多轮状态机。
因此在“次级候选无端到端收益即停止”的门槛处停止，没有把队列扩展到原发或生产路径。
该判断计入了已有队列微基准，不把隔离 sampler 吞吐误当成 histories/s。

## 推广决定与未完成精度项

没有候选满足 RT07575 至少+5%、原发和次级硬件指标同时改善的条件，所以没有获胜组合，
也无需运行只针对最终组合的 RT06423、20022516 和50k水模推广门槛。默认和回退均保持
现状；研究选项继续默认关闭。

本轮 GPU 自对照不代表患者精度验收。患者 RTSTRUCT BODY Gamma 和低密度
production-cut 阈值仍未完成；将来计算 Gamma 时必须只统计 BODY mask 内 voxel，且
非零 DTA 搜索步长使用 DTA/10。所有剂量验证继续使用3D scorer。

复现入口：`tools/freeze_transport_campaign.py`、`tools/benchmark_single_gpu.py`、
`tools/analyze_transport_campaign.py`、`tools/profile_transport_stages.py`、
`tools/summarize_transport_profiles.py`、`tools/summarize_transport_candidates.py` 和
`tools/assemble_transport_report.py`。统一机器报告为
`benchmark/transport_concurrency_20260916/campaign_report.json`。
