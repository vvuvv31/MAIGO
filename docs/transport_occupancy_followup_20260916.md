# 输运驻留与就绪度后续实验（2026-09-16）

## 结论

本轮分别处理原发驻留、次级路径上界和164-register次级的等待归因。所有候选继续默认关闭。

原发生产路径特化把实际sm_75入口从170降到162 registers/thread，十次RT07575配对中原发
kernel中位改善13.83%，Elapsed中位改善5.70%。但候选3D剂量最大差为峰值的
`0.00005684%`，超过十次控制重复包络`0.00004618%`。差异约为一个FP32 dose原子ULP；全部
histories、steps、核反应、统一EM整数审计、能量质量和overflow仍一致。按照严格包络门槛，
候选停止推广，不进入默认组合。

次级compile-only上界表明，known-species路径删除generic recoil后实际wrapper从164降到155
registers/thread；再删除non-He4路径的`he4_audit[6]`及三点rate审计后降到145。两个方向都
命中live-state高水位，但145仍没有到约128的下一驻留档。现有grouping只生成一份全局排序，
没有按bucket独立的continuation状态和压紧边界，因此不能把这两个probe直接用于生产调度。

对当前164-register实际wrapper重新采样后，旧版closure物化已不再是主要long-scoreboard来源。
最重采样点集中在52 B UnifiedEmNode和24 B cubic segment记录的二分搜索加载之后，另有少量
resume栈`LDL`消费者。下一轮应先测warp内EM node/segment查询键的unique数量；只有重复率足够高
才实现subgroup协作加载。raw local sectors不能直接解释为spill或独立耗时占比。

## 原发生产路径特化

`CARBON_PRIMARY_PRODUCTION_SPECIALIZE`默认关闭。host仅在统一EM、Schneider CT、3D voxel、
非弹性、多重散射、CT材料MCS和Schneider rate/stopping均满足固定生产组合，且all-elastic、
voxel额外clamp、charged-origin、primary-fluence与primary-loss-query audit均关闭时，调度
`CarbonPrimaryTransportKernel<1,true>`；其他配置保留通用回退。

|实际入口|registers/thread|stack/thread|
|---|---:|---:|
|控制 `CarbonPrimaryTransportKernel<1,false>`|170|2864 B|
|候选 `CarbonPrimaryTransportKernel<1,true>`|162|2848 B|

十次正式配对使用两次预热和交错正反顺序，未挂profiler。改善定义为
`control/candidate - 1`。

|指标|中位改善|十对范围|
|---|---:|---:|
|进程wall|+5.20%|+4.56% – +5.90%|
|Elapsed|+5.70%|+5.00% – +6.65%|
|原发kernel|+13.83%|+13.42% – +15.07%|
|次级kernel|+0.48%|−0.85% – +1.69%|

每次均为3,240,963 histories、1,034,976,717 steps和1,344,312次核相互作用；八项统一EM
整数审计一致、quality通过、overflow为0。十次控制包络没有覆盖候选3D dose，所以性能与资源
结果只作为失败归因证据，不能授权推广。

## 次级路径上界

两个开关都只用于compile-only上界，默认关闭：

- `CARBON_SECONDARY_KNOWN_SPECIES_PROBE`令生产特化实例中的`generic_recoil`编译期为false，
  wrapper为155 registers/thread、352 B stack。
- `CARBON_SECONDARY_NON_HE4_PROBE`在前者基础上删除He-4 hazard audit的恢复、逐步累计、保存与
  终止flush，wrapper为145 registers/thread、352 B stack。

固定RT07575中`use_all_elastic=false`，所以generic recoil本来不可达；non-He4 probe则只用于
资源上界，不能处理He-4分组。完整运行的输运整数审计保持一致，但这些probe没有生产安全分流，
没有进行正式配对，也不得作为用户配置或默认路径。

安全实现必须复用species×16 energy bucket排序，并为以下范围分别维护generation begin/end、
resume state、active order和压紧结果：non-He4 known species、He-4、unknown/recoil。每个粒子仍需
保持原队列索引与RNG身份；在完成该调度重构之前保持probe关闭。

## 164-register次级重新归因

NCU使用独立运行、kernel replay、`cache-control=all`和实际rounded wrapper；正式性能计时不挂
profiler。下表每阶段一次采样，active warp/SM由每scheduler数乘4得到。

|阶段|block|warp/SM|eligible/scheduler|ready fraction|long scoreboard|local sectors/(global+local)|
|---|---:|---:|---:|---:|---:|---:|
|早段|32|11.57|0.1218|4.21%|9.87|29.8%|
|中段|64|10.74|0.1050|3.91%|14.62|30.7%|
|晚段|128|4.01|0.0796|7.95%|4.05|45.4%|
|第二代|64|10.68|0.1090|4.08%|13.66|30.1%|

早段SourceCounters中两个最大long-scoreboard消费者紧随`LD.E.SYS`且以52 B stride访问记录；
另外四个主要消费者以24 B stride访问记录，分别对应Unified EM node与cubic segment布局。
这些依赖消费者合计占早段long-scoreboard PC samples的主要部分。另一个热点序列为
`LDL [R1+0x88] -> PRMT -> table lookup consumer`，说明剩余线程私有状态仍需定位，但它已不是
旧版约2 KB closure逐字节物化。

本轮application replay、`cache-control=none`只完成原发样本后因单阶段约4分钟而中止；该样本
与kernel-replay结果分开保留，不交叉比较。患者BODY Gamma和低密度production-cut阈值精度
验收仍未完成；GPU自对照不代表患者精度验收。

机器可读汇总见
`benchmark/transport_occupancy_followup_20260916/analysis.json`。
