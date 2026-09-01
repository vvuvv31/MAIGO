# 03B-2B：Bounded source/compiler consistency cleanup

## 目标

在 03A attribution sanity check 通过后，只修复实际 replay miss 的 source/compiler/package
一致性问题。该步骤仅覆盖 transportable isotopes；6Be 已是 `TopasCompatKill`，不参与 rate/replay
coverage。它是 correctness cleanup，不以 Be/Li 积分 dose 改善为验收。

## 固定输入

- canonical profile：c2e25b8 compatibility=true，200 MeV/u、100k、seed `2026095100`。
- 03B-2A audit：`plan/artifacts/cinel02-replay-support-occupancy-e200-g1-topascompat/audit.json`。
- 实际占用 miss：192；provisional raw/compiler gap：174；provisional index/lookup anomaly：18。
- Li6/Li7、Be7/Be9/Be10 的 occupied replay miss 当前均为 0。

## 工作范围

1. 对 174 个 gap 逐项回溯 raw exposure → compiler 输入/manifest → package event/support，
   确认是 raw source 真缺还是 compiler 丢 support；保存可复核的 provenance。
2. 对 18 个 anomaly 做 exact-energy replay，检查 global index、window selection 和 `(Z,A)`
   映射；为确认的 index/lookup bug 增加 synthetic regression。
3. 若 raw 有 support 而 compiler 丢失，优先修 source/compiler/package manifest；若 raw 真缺，
   只做显式缺口记录和 suppressed-hazard 评估，不静默改变 H/O target mix。
4. 不在 `E_rate` 上直接乘 support mask，不扩大 `±0.51 MeV/u` tolerance，不使用 nearest-event
   fallback；只有 occupancy 证据证明 `E_rate → E_replay` 边界漂移占主导时，才另立 segmentation
   设计。

## 退出条件

- [ ] 174 个 gap 和 18 个 anomaly 均有 raw/compiler/package 分类证据。
- [ ] transportable isotope replay miss 接近 0（或每个保留缺口都有显式 suppressed hazard 记账）。
- [ ] 修复前后 package/rate hash、target mix、birth count/KE expectation 无可测重归一化。
- [ ] C++/Python regression 覆盖 exact-energy lookup、source/compiler manifest 和 H/O/G1/G2。
- [ ] 未以 integrated dose 改善或 Be/Li 误差下降作为通过条件。

## 完成后总结

完成本步后立即转入 `track-length × hazard optical depth → empirical reaction survival → stopping residence`。
lookup miss 不再阻塞 Be/Li 主线，除非出现 transportable Be/Li 的直接 support 缺口证据。
