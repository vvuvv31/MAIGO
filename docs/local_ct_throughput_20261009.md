# 本地 CT 宽束吞吐优化 — 2026-10-09

L4 通过验证的源码已同步到本地，并重新编译。L4 原始证据完成校验后已关闭。此次仅研究质子/碳离子 RT07575 head CT 宽束，水的性能优化暂缓。

本地硬件 RTX 2080 Ti（sm_75），driver 560.35.05。对照 128/256 主粒子工作组，源文件仅一行差异。每进程连续 4×1,000,000 主粒子，复用 GPU 上下文/数据包，舍弃首批。全 campaign 最大余数分配、spot seeds、物理模型和归一化保持一致。设置 CUDA_DISABLE_PTX_JIT=1，运行成功。无法锁频，记录前后实际频率/温度/功率并使用交错比较。

## 首轮 ABBA

| 配置 | 128线程 粒子/s | 256线程 粒子/s | 256线程提升 |
|---|---:|---:|---:|
| proton_ct_broadbeam | 143,415 | 151,061 | 5.33% |
| carbon_ct_broadbeam | 69,975 | 72,294 | 3.31% |

两对比较方向均支持 256 线程；结果仍需反向 BABA 重现，不能把未锁频单轮百分比当作固定硬件常数。匹配剂量 local1% 零距离通过率均 100%（参考剂量≥10%全局峰值），作为 local1%/1mm 的通过率下界；离散/源/seed/能量/EM覆盖/完成/零队列溢出审计通过。这是 GPU 实现匹配检查，不是新 TOPAS 验收。

反向 BABA 与碳 CT 宽束 CUDA 时间线均已完成，质子 CUDA 时间线也已补充。反向测试重现：Proton +4.17%，Carbon +2.42%。详细瓶颈见 [local_ct_bottlenecks_20261009.md](local_ct_bottlenecks_20261009.md)。profiler 的扰动时序不纳入性能结果。状态及证据位于 `out/local_ct_throughput_20261009/`，包括 `campaign_status.json`、`done.json`、`reverse_done.json`（完成后生成）、`profile_carbon.nsys-rep`（完成后生成）。当前主代码保留已同步的 CT256 和此前电子 packet 原位更新；未推广额外缓存或实验接口。

首个启动检查因对照程序尚未完成链接而停在第一臂，该次数据保留在 runs/，不计入正式结果；完整正式 ABBA 位于 formal_runs/，编译结束后重新运行。
