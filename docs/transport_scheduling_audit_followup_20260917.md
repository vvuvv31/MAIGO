# 调度开销、EM单线程依赖链与He-4审计跟进（2026-09-17）

本轮以 `557bec194382f867824d82230dd5b71a116a3746` 为基线，在本地 RTX 2080 Ti / sm_75 上执行
五组实验，区分“调度/launch成本”和“kernel内部资源”。所有新候选默认关闭，唯一的默认改动是
He-4 sparse audit 搬出 long-lived 续跑状态（寄存器下降、审计值一致、无吞吐回归）。未改变
物理数据、RNG身份、16步续跑默认、8192尾阈值、精确CT边界或64路统一EM审计。Schneider v2.1
运行前16项manifest校验通过。患者BODY Gamma和低密度production-cut阈值精度仍未完成；本轮
quality通过不代表这两项精度已经验证。本轮未 commit / push。

## 1. 二级slice去同步（去`wait_and_throw`、去boundary kernel）

- 队列本身是`in_order + enable_profiling`，因此连续submit三个slice与逐个wait的执行顺序相同；
  改动只删除两次host-device往返。
- stable compaction是有序的，新proton/deuteron幸存数可直接由`block_offsets[i/256]+ranks[i]`
  在既有prefix single_task内得到，不再需要额外的bisection kernel。
- 资源不变：exact proton/deuteron 124/124 registers、stack 352 B；fallback slice 139 registers。
- 五次交错配对（改善定义为`control/candidate-1`）：

|指标|中位|五对范围|
|---|---:|---:|
|secondary|−3.79%|−4.17% – −3.75%|
|Elapsed|−1.66%|−2.02% – −1.28%|
|primary|+0.01%|−0.50% – +0.48%|

去同步把此前的次级退化从约 −4.47% 收回到 −3.79%（约0.7个百分点），但不足以抵消三路launch与
slice尾部成本。结论：仍失败，默认关闭。

## 2. proton/deuteron延长continuation budget（16→32/64）

fallback保持16步，仅exact p/d使用更长segment，期望把专用kernel的launch数降到约1/4。

|budget|gen1 rounds|secondary中位|Elapsed中位|
|---:|---:|---:|---:|
|16/16|36|−3.79%|−1.66%|
|64/64|30|−7.54%|−3.15%|

rounds只从36降到30，说明轮数由fallback物种决定；p/d提前退出后，剩余fallback launch的粒子数
更少、填充率更低，尾部成本反而上升。结论：budget加长更差，`CARBON_SECONDARY_HOT_SEGMENT_STEPS`
默认16保持原行为。

## 3. H(p+d) class编译探针

新增compile-only哨兵 `CARBON_SECONDARY_EXACT_SPECIES_PROBE=-2`（`H1_H2` class，species index
仅0/1，运行时z/a）。实际rounded wrapper为 **139 registers / 352 B stack**。

关键负结果：本轮He-4 sparse audit搬家后，通用fallback wrapper已从150降到136 registers，因此
H class的139寄存器不但未到≤128门槛，反而高于fallback。结论：不实现运行时分派，保持compile-only。

## 4. EM exact-index interval-width直方图

新增只读诊断`CARBON_EM_INTERVAL_WIDTH_AUDIT`，在真实`bounds()`中统计`end-begin`宽度。
完整RT07575固定分片：

|查询|总数|w0|w1|w2|w3|w≥4|
|---|---:|---:|---:|---:|---:|---:|
|node|7,162,724,353|0|0|0|0|7,162,724,353|
|raw0|4,932,618,916|0|0|0|0|4,932,618,916|
|raw1|4,932,618,916|0|0|0|0|4,932,618,916|

没有任何“一个bucket恰好一个knot”的场景，packed-pivot index省不掉dependent key load。结论：
不实现，默认关闭。

## 5. He-4 sparse audit搬出续跑状态（唯一默认改动）

把`he4_audit[3]`（inelastic候选计数）和`[4]`（候选能量）改为事件发生时直接累加global device
tally，per-track只保留0/1/2/5。设备槽位语义与`_he4_hazard.csv`表头保持一致
（`tau_start,tau_simpson,energy_tau_simpson,candidates,candidate_energy,path_mm`）。

|入口|基线 regs|本轮 regs|
|---|---:|---:|
|production fallback (`...ILi1ELb1ELin1E`)|150|**136**|
|exact He-4 (`...ILi1ELb1ELi4E`)|130|**127**|
|exact proton/deuteron|124|124|
|stack|352 B|352 B|

- 吞吐：五次交错配对 secondary中位 **+0.008%**、Elapsed −0.03%、primary +0.25%，全部在重复波动内。
  即150→136的寄存器下降没有转化为吞吐，符合此前“该内核ready fraction约4–5%，不是驻留受限”的判断。
- 正确性：开启`fragment_birth_spectrum_output_file`后，He-4 hazard审计`candidates=19088`与基线
  完全一致，其余槽位仅浮点累加顺序差异（相对~1e-14）。固定分片八项统一EM整数审计一致、
  quality通过、overflow=0。
- 结论：作为寄存器清理保留默认开启；不构成吞吐推广。He-4专用occupancy验证仍被“无可行低launch
  调度”阻塞——hot species移除后，He-4的低寄存器优势没有运行时路径。

## 正确性门槛

- 固定RT07575分片：3,240,963 histories、1,344,312次核反应；八项统一EM整数审计
  `0 1036320939 1429988519 0 0 6897079514135703 528848467657861 0` 与基线逐项一致。
- quality=pass、queue overflow=0、能量相对残差 `3.68334338772e-06`（基线 `3.6833317e-06` 量级）。
- Schneider v2.1全部16项manifest与固定哈希通过。
- 未做患者BODY Gamma；本轮为GPU自对照，不代表患者精度验收。

## 复现

无profler配对使用`tools/run_transport_candidate_pairs.py`，2次预热、5次交错正式配对；
构建为sm_75 Release，`CARBON_DOSE_FP32=ON`。候选构建：

- `CARBON_SECONDARY_CONTEXT_POINTER=ON`、`CARBON_SECONDARY_PRODUCTION_SPECIALIZE=ON`（控制组）；
- 加`CARBON_SECONDARY_HOT_SPECIES_SPECIALIZE=ON`与`CARBON_SECONDARY_HOT_SEGMENT_STEPS=16|32|64`（候选1/2）；
- 加`CARBON_SECONDARY_KNOWN_SPECIES_PROBE=ON -DCARBON_SECONDARY_EXACT_SPECIES_PROBE=-2`（H class探针）；
- 加`-DCARBON_EM_INTERVAL_WIDTH_AUDIT=ON`（直方图）。

机器可读结果见`benchmark/transport_scheduling_audit_followup_20260917/analysis.json`。
