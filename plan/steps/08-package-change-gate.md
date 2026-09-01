# 08：Sampler/Package 修改门禁

## 目标

只在 TOPAS 高统计和 deterministic auditor 同时指向 package construction 时，才允许修改局部 support/package。

## 决策矩阵

1. job 485 discrepancy 随统计量消失，且 package expectation 与 source 一致：永久冻结 sampler。
2. job 485 仍偏，package expectation 与 source 一致：检查 full-cascade parent occupancy、target selection、energy mapping，不改 sampler。
3. package expectation 本身与 source 不一致：检查 duplicate event、energy boundary、node/event weight、campaign merge、O16 identity、hybrid weighting。
4. 只修发现的 construction cause，然后重做 exact expectation 和 fixed-seed GPU A/B。

## 允许的修改

- compiler deduplication/boundary/support metadata 错误。
- 事件权重与 source exposure 不一致的可证明修复。
- 身份、target 或 campaign provenance 错误。

## 禁止的修改

- `Be9_scale`、global Be normalization、energy-specific dose scale。
- 恢复 nearest-single event、任意扩大 tolerance、跨 gap 最近事件。
- 通过 MCS/stopping/straggling 补偿错误 yield。

## 退出条件

- [ ] job 485 与 deterministic auditor 结论同向。
- [ ] 修改原因对应具体 `(projectile,target,Ebin,channel)`。
- [ ] 修复后 runtime actual、package expectation、TOPAS source 统计相容。
- [ ] 其他 isotope/channel 无回归。
