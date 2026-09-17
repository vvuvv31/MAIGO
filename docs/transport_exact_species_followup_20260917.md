# Exact-index与物种特化并发跟进（2026-09-17）

本轮以`dd2c6d0`和RTX 2080 Ti/sm_75为基线，继续区分驻留warp与单warp readiness。所有新增候选和诊断开关默认关闭；未改变用户配置格式、物理数据、RNG身份、16步续跑、8192尾阈值、精确CT边界或64路统一EM审计。Schneider v2.1运行前16项manifest校验通过。患者BODY Gamma和低密度production-cut阈值精度仍未完成。

## Exact-index空桶直达

`CARBON_EM_EMPTY_BUCKET_FAST_PATH`在指数桶的`begin==end`时令`hi=lo`，不读取node/segment key；边界等号、插值和运算顺序不变。独立`CARBON_EM_EMPTY_BUCKET_AUDIT`在完整RT07575中记录到：

|查询|总数|空桶|命中率|
|---|---:|---:|---:|
|node|7,162,724,353|0|0%|
|raw0|4,932,618,916|0|0%|
|raw1|4,932,618,916|0|0%|
|raw2|530,652,722|0|0%|
|raw3|0|0|—|

fast path仍通过82,741,572个GPU节点/边界相邻浮点查询，逐位失败为0，但真实输运无动态命中。因此停止性能计时和NCU，默认关闭。

## 物种步数与compile-only下界

`CARBON_SECONDARY_STEP_PROFILE`测得前三个物种为proton 557,526,724步、He-4 364,710,654步、deuteron 281,766,834步。exact-species probe将`(Z,A)`、charged/EM species index、质量倒数和表偏移编译期化，并在生产host guard确认all-elastic、unified-water和charged-origin关闭后删除对应路径。

|路径|实际rounded wrapper registers/thread|stack/thread|
|---|---:|---:|
|原production wrapper|164|352 B|
|仅固定生产模式的fallback|150|352 B|
|exact proton|124–125|352 B|
|exact deuteron|123–124|352 B|
|exact He-4|130|352 B|

proton和deuteron跨过约128档，He-4没有。因此只实现proton/deuteron运行候选；其余物种和He-4走通用fallback。

## 稳定slice调度

`CARBON_SECONDARY_HOT_SPECIES_SPECIALIZE`复用现有species×16 energy bucket排序。resume state和active order仍共享；stable scan/scatter后，通过有序state index二分得到proton/deuteron survivor边界。每轮依次启动proton、deuteron和fallback slice，粒子原队列索引、RNG stream、暂停/恢复和审计身份不变。

完整固定分片通过3,240,963 histories、1,034,976,717 steps、1,344,312次核反应，八项统一EM整数审计一致、quality通过、overflow为0。但两次预热和五次交错配对中，改善定义为`control/candidate-1`：

|指标|中位变化|五对范围|
|---|---:|---:|
|wall|−1.71%|−1.86% – −0.87%|
|Elapsed|−1.86%|−2.03% – −1.01%|
|primary|+0.16%|−0.50% – +0.60%|
|secondary|−4.47%|−4.65% – −3.24%|

三路launch和slice尾部成本超过低寄存器收益。候选最大3D剂量差为峰值0.00004618%，也略超过本轮五次控制包络0.00003908%。该调度默认关闭并记入失败台账。

## 固定生产模式的独立结果

移除slice调度后，生产host guard仍可将all-elastic、unified-water和charged-origin确定为false，使fallback从164降到150 registers。五次配对结果：secondary中位提高4.46%（4.06%–5.06%），Elapsed提高1.95%（1.77%–2.35%），3D剂量差不超过控制包络。

匹配早段NCU显示block 32、150 registers、11.50 warps/SM、eligible 0.1241/scheduler、ready fraction 4.31%。164-register基线分别为11.57、0.1218和4.21%。驻留没有跨档，eligible小幅提高，但long-scoreboard从9.87升至11.94。该项是正向次级候选，但端到端不足5%且硬件组合门槛未通过，继续默认关闭，不做跨病例推广。

## 搜索地址重复度与pair prepare

`CARBON_EM_SEARCH_KEY_AUDIT`采样实际rounded wrapper中前4,000,000条dependent comparison地址；分组键为launch、warp、step、query ordinal和kind。node/raw0/raw1平均各约18.4个唯一地址，约42k组中只有4–6组不超过4个唯一地址。raw2平均7.72个唯一地址，仅15,887组，其中2,957组不超过4个。该重复度不足以支付ballot/shuffle和额外控制流，因此没有实现subgroup cooperative load。

`CARBON_EM_PAIR_PREPARE`先物化lo/hi point再执行两个prepare。实际wrapper仍为150 registers，但单次完整筛选secondary从7.9114 s退化到8.1555 s，Elapsed从19.7554 s退化到20.0897 s，停止正式配对并默认关闭。

## 原发同一二进制机制诊断

`CARBON_PRIMARY_PATH_DIAGNOSTIC=generic|specialized`只在包含两实例的诊断构建中选择入口，确保二进制哈希相同。五次交错配对再次得到primary中位提高13.76%、Elapsed提高5.97%，但specialized最大3D剂量差为峰值0.00005684%，超过generic五次包络0.00004263%。

FP64 scorer仅用于机制诊断：同一二进制generic与specialized的最终3D输出逐位相同，最大差0、非零差voxel为0；输运整数审计也一致。这确认FP32门槛失败来自occupancy改变后的原子累加顺序，而不是物理/RNG路径变化。既定FP32严格包络没有因此豁免，原发特化仍默认关闭。

机器可读结果见`benchmark/transport_exact_species_followup_20260917/analysis.json`。无profiler的
交错配对可用`tools/run_transport_candidate_pairs.py`复现；脚本拒绝覆盖已有输出目录，并记录每次
wall、Elapsed、原发/次级时间、温度和SM时钟。
