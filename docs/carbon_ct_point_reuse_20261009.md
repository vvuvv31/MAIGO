# Carbon CT broadbeam 同步 EM 视图复用 — 2026-10-09

本轮保留同一步计算复用；尚未达到 proton broadbeam 的吞吐水平。RTX 2080 Ti、sm_75，本地原 RT07575 **头部** CT carbon broadbeam 计划。未改水或 minibeam 默认配置。

## 实现

`src/detail/sycl_pipeline_primary.inc` 将已有 proton CT stage13 的同一步 continuous26 视图复用和精简 electron birth stopping 查询扩展到 carbon CT stage14。`src/detail/sycl_pipeline_secondary.inc` 仅在 CarbonBroadbeam 的 CT 专用 secondary launch 中复用同一步 EM point 的 preparation 和 loss evaluation。

没有跨步缓存，没有扩大 continuation state。保留全部 charged fragment species、C12 A*amu 运动学、continuous26 校准公式、涨落、核反应、步长上限、电子/光子、FP32 dose atomics、source/spot seeds 和归一化。其他 pipeline 的 secondary 继续原 evaluator。

## 验证与正式性能

CPU checker 覆盖 18 种 package species ×26材料 ×7密度 ×100能量 ×4步长 ×2涨落，共2,620,800个条件。prepared字段、birth stopping/delta、loss字段和随机数状态精确一致，invalid=0。见 `out/carbon_ct_point_reuse_20261009/check_all_species.cpp` 和 JSON。

仅编译 carbon broadbeam，独立目录 `build-carbon-broadbeam`。基线和候选均由同一工具链及 CMake 开关构建；禁用 PTX JIT 成功执行，已检查 sm_75 cubin。ABBA 和独立种子 BABA，每进程4×1M粒子、连续复用GPU context/packages，性能剔除首批。两组总共32M粒子，每臂16M，其中暖批12M。

| 对照 | candidate/control 吞吐比 | 两个配对比 |
|---|---:|---|
| ABBA | 1.031255 | 1.014341 / 1.047888 |
| BABA | 1.039020 | 1.048071 / 1.030135 |
| 合计 | 1.035174 | control 72697.3/s → candidate 75254.4/s |

四个配对方向一致，合计+3.52%。GPU不能锁频，测试存在热漂移，百分比是本轮观测结果。

全部 source、离散事件、live ion species ledger 和整数EM/charged/photon/joint审计匹配；energy、coverage、completion 和 zero-overflow通过。匹配剂量 relative L1≤4.30952502e-08，10% peak以上体素同位置local1%通过率100%，因此local1%/1mm、search0.1mm门槛的通过率下界100%。这是GPU实现一致性检查，没有新的绝对TOPAS验收。

## 为什么尚未达到质子水平

质子最近独立测试约153,769/s，本轮碳束约75,254/s，仍相差约2.04倍。这是各自计划的原初粒子/s比较，未证明相同剂量统计误差下的效率比例，质子也未在本轮同时重跑。

当前模型原始profile中碳primary3.6629s/M、secondary6.4561s/M、electron3.5892s/M。碳的主要额外工作是带电碎片及其下游输运。若primary不变，仅优化碎片即使完全消除其耗时，剩余primary+electron也约7.25s/M，尚高于质子约6.50s/M；此为历史profile的Amdahl估算，不是本轮候选的新profile。若要达到目标，需要同时大幅优化secondary和electron，不能由本轮+3.52%外推出翻倍。

下一步优先按碎片物种、能量、代数归因真实输运成本，再考虑专用输运或带权碎片抽样。带权方案需把weight贯穿所有后代及计分，以达到同等误差的时间验收；直接关闭碎片、电子或局部沉积其动能不构成本任务的等物理优化。

原始证据：`out/carbon_ct_point_reuse_20261009/`，包括control/candidate源码、binary、host checker、ABBA/BABA命令/日志/逐批剂量/审计和summary。

最终说明注释更新后重新编译，全部 native SASS 与正式测试候选完全一致；见 `final_sass_check.json`。
