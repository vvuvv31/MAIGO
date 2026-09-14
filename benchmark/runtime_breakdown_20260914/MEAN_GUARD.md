# 原发重复平均能损：隔离 A/B

## 发现与改动

深层 runtime 采样将此前混合的原发准备区分为：几何截步 3.90%、核率/光学深度 8.22%、外层平均能损/审计区 20.41%。这些是诊断周期比例，不是独立墙钟。

冻结源码先调用 `unified_primary_state.mean(T,h)`，随后 `unified_em_loss()` 内部再计算物理平均能损。统一 EM 且 `enable_primary_loss_query_audit=false` 时，外层结果在最终抽样前被覆盖，未用于最终输运。

隔离候选只让外层查询在 `primary_loss_query_audit` 非空时执行。实际能损、涨落、δ 时钟、RNG、MCS、核过程和记分算法保留。未采用任何物理近似；诊断打开时仍执行原来的查询。

候选位于 `scratch/runtime_breakdown_20260914/mean_guard_source`；补丁 `mean_guard.patch` 相对于此次冻结工作树，不保证可直接应用于任意 HEAD。正式源码尚未接入。

## 百万粒子同配置对照

当前生产入口，完整次级统一 EM、种类分组、无独立核弹性、chunk=34816。第一轮基线为本次 profiling 前未插桩基线；后续顺序为候选1、基线2、候选2。二进制由相同编译选项生成。

| 轮次 | 基线 histories/s | 候选 histories/s | 提升 | 原发 kernel 基线→候选 |
|---|---:|---:|---:|---:|
| 1 | 19585.9 | 21745.6 | 11.03% | 25.595 → 20.574 s |
| 2 | 19243.7 | 21156.0 | 9.94% | 26.102 → 21.301 s |

两轮 histories/s 均值：19414.8 → 21450.8，提高 10.49%。次级耗时基本不变；全进程 wall 和源/二进制 SHA 见 mean_guard_results.json。

全部比较的步数、EM audit、核反应/overflow 计数一致，最大体素剂量差为峰值的 0.00002657%。均执行质量接受，零 overflow。

另以 200k 原发、smoke 模式启用 `enable_primary_loss_query_audit=true`：基线与候选的 14 个 PRIMARY_LOSS_QUERY 桶计数和能量整数和完全一致，剂量等价检查通过。该测试用于保证诊断功能未被删除，不计作提速证据。

## 结论

这是已通过局部 A/B 的可消除重复计算，不应归入几何或核事件成本。50k/s 尚未达到；修正后原发仍约 21 秒、次级约 22 秒，继续优化需覆盖两者。次级热点为原生 stopping/range/涨落参数准备及实际 EM 抽样。

原发核率缓存之前仍执行一次 primary_rates()，可能存在进一步避免无效查询的空间，但本次没有修改该行为，也未验证对应加速。

复现运行：`python3 benchmark/runtime_breakdown_20260914/run_mean_guard.py`；检查：`python3 benchmark/runtime_breakdown_20260914/compare_mean_guard.py`。配置依赖前一轮固定 inputs 和源码快照。

这项计算冗余优化的验证不等于解决种类分组在含弹性 1M 情景中的 0.0203% 剂量差异，也不升级整体患者 Gamma 验收。
