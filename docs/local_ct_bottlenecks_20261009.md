# RT07575 head CT broadbeam 性能瓶颈 — 2026-10-09

使用当前本地 release（含 CT256 和电子原位 packet 更新），RTX 2080 Ti / sm_75 / driver 560.35.05。质子与碳离子分别采集 CUDA 时间线：每进程四批，每批一百万主粒子。GPU 统计剔除首批，取第二批第一个 primary kernel 开始后的三个完整批次。profiler 时序只用于瓶颈定位，不替代不带 profiler 的正式吞吐结果；primary/secondary/electron 为串行 GPU kernel 分组。

## 实测 GPU 内核分解

| 路径 | Proton 秒/百万主粒子 | GPU时间占比 | Carbon 秒/百万主粒子 | GPU时间占比 |
|---|---:|---:|---:|---:|
| 主粒子离子 | 3.8187 | 60.15% | 3.6629 | 26.70% |
| 带电次级离子/碎片 | 0.2818 | 4.44% | 6.4561 | 47.05% |
| 独立电子/光子 worker | 2.2449 | 35.36% | 3.5892 | 26.16% |
| 整理/其他内核 | 0.0037 | 0.06% | 0.0126 | 0.09% |

Proton 最大瓶颈是主粒子内核，Carbon 最大瓶颈是带电次级碎片。两者主粒子内核时间相近，但 Carbon 次级内核约慢 23 倍；每百万主粒子生成约 418 万带电次级队列项，Proton 约 55 万。Carbon 电子推进工作量也更高。

非 profiler 的暖批：Proton 总 wall约 6.70s，setup约 0.20s，readback/host finalize约 0.042s；Carbon wall约 13.98s，setup约 0.109s，readback/host finalize约 0.042s。计时阶段存在嵌套，不能累加全部 runtime-stage。EM package 暖批 device upload约 0.00007s，上下文/包已复用。GPU copy 暖三批累计：Proton约 0.171s，Carbon约 0.117s，因此实际 PCIe 数据传输不是主要耗时。

CUDA API 表显示的长 cuMemcpyAsync/cuEventSynchronize 时间包含等待此前排队 GPU 工作，不是独立传输耗时。Carbon 全四批 CUDA API 中 cuMemcpyAsync累计29.38s，但 GPU memcpy活动只0.36s；不能把 API 名字的占比误当成带宽瓶颈。

## 内核为何难以跑满计算资源

CUDA trace 确认当前实际 launch 主质子231、主碳234寄存器/线程，block256；电子 worker166寄存器/线程，block256。Turing每SM65536寄存器、最多1024驻留线程，在这些配置下主粒子/电子的寄存器约束仅允许一个256线程 block/SM，理论驻留上限约25%。这是资源上界，不是已测得的动态 occupancy或stall占比。静态 cubin中主粒子还有2384B call stack，secondary1776B左右；该资源信息本身不能证明 spill流量主导。带电次级内核实际202/204寄存器，运行 block大小随SYCL range/活跃队列变化。

碳束还有通用多物种 fragments、不同能量/射程/材料下的CT几何、continuous26查表、核反应和涨落状态；长寿命线程状态限制资源驻留，混合轨迹可能产生分歧和依赖访存等待。寄存器资源限制与工作量已确认，具体指令/访存/分歧贡献仍需硬件计数器或独立受控实现实验辨别。

Nsight Compute 已尝试收集 Occupancy、WarpStateStats、MemoryWorkloadAnalysis和SpeedOfLight；驱动拒绝 ERR_NVGPUCTRPERM，且sudo不可无交互提权。保留失败证据，不把旧版本动态profile或静态寄存器推断当作当前SASS级热点实测。

## 调度成本和优化优先级

Carbon每百万主粒子带电次级86轮，尾部约0.106s，compaction约0.011s；Proton43轮，尾部约0.041s，compaction约0.0028s。Carbon物种分组约0.003s。队列整理只占极小部分；主要成本是被调度的输运工作本身。首次使用包/模块冷启动不计入稳态吞吐，已验证CUDA_DISABLE_PTX_JIT=1成功执行，当前低稳态吞吐不能归因于JIT。

后续顺序：Carbon优先研究带电碎片 worker的热点状态/查表/几何与低开销调度；Proton优先主粒子热点状态，其次两者共享的电子 worker。所有路径保持物种、物理模型、FP32 dose、每spot种子、能量/覆盖/完成/零溢出审计和匹配剂量约束。不能用删掉物理工作来宣称实现吞吐提升。

以Carbon GPU占比估算，即使把primary耗时减少20%，整体GPU时间只减少约5.3%；secondary耗时减少20%则整体约减少9.4%。这是Amdahl比例示例，不是已实现性能收益。此前分组/compaction/阈值、强压寄存器和缓存失败记录仍有效，不原样重试。

当前同步的CT256优化已在两种交错顺序重现：Proton +5.33% / +4.17%，Carbon +3.31% / +2.42%；各匹配剂量/源/离散/能量/EM/完成/零溢出检查通过。无法锁频，因此百分比是这轮本机观测值，不是固定跨硬件常数。没有新的TOPAS验收结论。

证据目录 `out/local_ct_throughput_20261009/`：`bottlenecks.json`、`stage_analysis.json`、两种粒子的 `profile_*.nsys-rep`/SQLite、`kernel_resources.txt`、`ncu_carbon.csv`、正式ABBA/BABA结果。L4保持关闭。
