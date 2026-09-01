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

## 工作范围与结果

1. [x] 对 174 个 provisional gap 按 raw exposure → compiler 输入/manifest → package
   event/support 做 provenance census。3,663,101 条 raw record 与 package 的 3,663,101 条
   interaction 全部匹配；36 个 isotope×target key 中 34 个为 `source_present_compiled`，
   2 个为预期的 6Be `source_missing`；`compiler_dropped_support=0`、
   `package_has_unbacked_support=0`、raw/package 不匹配均为 0。20 个 provisional gap cells
   全部落在 `source_present_compiled` key，故不是 compiler 丢包，而是运行时稀疏 support
   occupancy gap。
2. [x] 对 18 个 provisional anomaly 做 exact-energy replay/index probes。package 的
   3,573,494 个 global energy nodes 共执行 14,293,976 个 node/window 边界查询，
   `exact_replay_failures=0`；因此没有确认的 global-index、window-selection 或 `(Z,A)`
   lookup bug，9 个 anomaly cells 保留为运行时 provisional 分类。
3. [x] 生产 package C++ inspection 与 Python census 交叉一致：34 个 source-backed
   projectile/target groups、3,663,101 interactions、30,051,236 products、
   3,573,494 nodes；6Be 不作为 transportable projectile 编译。
4. [x] 不改变 package/rate、target CDF、`±0.51 MeV/u` tolerance、isotope yield 或
   runtime rate mask；没有 source/compiler 修复项，也没有 dose-driven 重归一化。

## 结论

03B-2A 的 192 个 miss（174 个 provisional gap miss、18 个 provisional anomaly miss）
没有得到 source/compiler/package 或 global-index 的确认缺陷证据。它们应作为 runtime
稀疏 support 的 correctness residual 携带，而不是通过 nearest-event、放宽 tolerance、
改变 H/O target mix 或静默 suppress hazard 来修复。由于 Li6/Li7、Be7/Be9/Be10 的实际
miss 均为 0，03B-2B 不再阻塞 Be/Li 剂量主线。

## 测试与退出条件

- [x] Python synthetic regression 覆盖显式 isotope/target key 表、replay tolerance、
  exact node/window boundary 和 `(Z,A)` key mismatch；既有 C++ null-collision/MCS
  regression 保持通过。
- [x] raw/compiler/package provenance、H/O key 和 0/1/2 `reaction_generation` 的生产
  census 已完成；没有需要修改的 source/compiler manifest。
- [x] transportable Be/Li replay miss 已实测为 0；其余 192 个 miss 有 provisional
  classification、source/package provenance 与 exact-probe 结果，并显式携带到后续
  runtime correctness 评估，不以 dose 改善作为通过条件。
- [x] 未扩大 tolerance、使用 nearest-event fallback、在 `E_rate` 上静默乘 runtime
  support mask，也未改变 rate/target mix 或 birth expectation。

## 完成后总结

本步完成了 bounded source/compiler consistency audit：证明当前 192 个 replay miss 不是
compiler 丢 support 或 global-index 错误，且没有需要提交的 physics/package 修改。下一步立即
转入 transportable Be7/9/10、Li6/7 的 `track-length × hazard optical depth → empirical
reaction survival → stopping residence`；03B-2B 不再阻塞该主线。
