# 次级生产路径特化实验（2026-09-16）

## 结论

在隔离的`b3b906d`源码上，生产路径host-dispatch特化把实际range wrapper从177降到164
registers/thread，首次跨过sm_75的168寄存器驻留档位。RT07575五次正式配对中，次级kernel
中位加速9.69%，端到端Elapsed中位改善3.92%。候选通过结构门槛和正确性门槛，可以进入后续
组合，但单项端到端收益没有达到5%默认推广门槛，因此
`CARBON_SECONDARY_PRODUCTION_SPECIALIZE`保持默认关闭。

该结果修正了混杂研究工作树中的初步判断。只有排除dense/shared/fine EM和步长研究改动后，
实际提交源码才出现164-register产物；提交和报告以这个隔离源码及固定二进制哈希为准。

## 实现

host仅在`EmMode == 1`且统一次级EM、CT网格、3D voxel计分、精确CT面边界和同材料face
clamp跳过均启用时调度`CarbonSecondaryTransportKernel<1, true>`。同一二进制保留
`CarbonSecondaryTransportKernel<EmMode, false>`回退。generic recoil、核反应、RNG、剂量
记分和续跑顺序均保留；开关要求`CARBON_SECONDARY_CONTEXT_POINTER=ON`。

|实际range wrapper|registers/thread|stack/thread|constant0|
|---|---:|---:|---:|
|context通用实例|177|352 B|408 B|
|生产特化实例|164|352 B|408 B|

## 正式配对性能

RTX 2080 Ti/sm_75，预热2次，5次正反交错配对，不挂profiler。改善定义为
`control/candidate - 1`。

|指标|中位改善|五对范围|
|---|---:|---:|
|进程wall|+3.63%|+3.16% – +4.22%|
|Elapsed|+3.92%|+3.41% – +4.59%|
|原发kernel|−0.11%|−0.60% – +0.46%|
|次级kernel|+9.69%|+8.71% – +10.53%|

候选对原发没有有意义的影响。端到端收益方向稳定且超过本轮波动，但低于5%单项默认推广线。

## 匹配阶段硬件结果

使用独立NCU运行，按代次、续跑轮次和active粒子数匹配。下表是一次匹配样本；正式推广仍需
重复采样。active warp/SM由每scheduler数乘4得到。

|阶段|active warp/SM control→candidate|eligible/scheduler|ready fraction|long scoreboard|
|---|---:|---:|---:|---:|
|次级早段|7.76→11.51|0.0993→0.1209|5.12%→4.20%|6.87→9.98|
|次级中段|7.83→10.65|0.0950→0.1049|4.85%→3.94%|12.15→14.88|
|次级晚段|4.00→4.00|0.0798→0.0795|7.98%→7.94%|4.13→4.09|
|第二代早段|7.74→10.61|0.0978→0.1099|5.06%→4.14%|11.23→13.44|

稳态阶段实际驻留量和eligible均提高，证明164 registers确实跨过驻留档位。ready fraction下降、
long-scoreboard上升，说明新增warp主要提供更多latency hiding，没有减少每个warp自身的等待。
晚段仍由10,533个活跃粒子的tail underfill限制。

## 正确性与采用状态

每次均为3,240,963 histories、1,034,976,717 steps、1,344,312次核相互作用；整数审计一致，
quality通过且overflow为0。候选最大3D剂量均值差为峰值的0.00003908%，没有超过本轮基线重复
包络0.00003908%。Schneider v2.1、统一EM、delta moments、`straggling_scale=1`、精确CT
边界、16步续跑和8192尾部阈值保持不变。

候选保留默认关闭，下一步是与context-pointer一起完成A6000和跨病例推广，并重复匹配NCU。
最终组合仍需满足至少5%端到端收益及既定硬件门槛。患者BODY Gamma和低密度production-cut
阈值精度验收仍未完成；GPU自对照不代表患者精度验收。

机器可读结果见`benchmark/secondary_production_specialization_20260916/analysis.json`。
