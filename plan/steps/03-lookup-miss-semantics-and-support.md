# 03：Lookup Miss Semantics 与 Event Support

## 目标

先保证 replay miss 不扭曲 EM/MCS transport，再使 rate coverage 与 event support 严格一致。

## 问题

rate group 可在连续能区插值为非零，但 event replay 要求当前 `±0.51 MeV/u` 真正有 event。200 MeV/u G1 已观测 `192/33512 = 0.573%` secondary misses。

## 阶段 A：Null-collision transport

1. miss 保留已走路径的 continuous EM loss 和 post-EM state。
2. miss 也必须执行与普通步一致的 MCS/lateral displacement，不得人工直线飞行。
3. 不产生 products、local nuclear deposit 或 parent final-state 变化。
4. 记录 `(Z,A,target,E,generation,depth)` miss histogram 和 associated path/KE。

## 阶段 B：Support-aware rate

1. compiler/manifest 为每个 rate group 输出真正 event-support intervals/mask。
2. runtime 仅在 tolerance window 内存在 event 时标记 replay-covered。
3. 不跨大能差寻找 nearest event，不扩大 tolerance 来隐藏 gap。
4. 检查 support mask 对 rate/reaction CDF 的影响，不允许无声重归一化 isotope yield。

## 测试

- 合成 gap：rate-covered/event-uncovered、boundary hit、H/O 切换、G1/G2。
- miss 步的 energy、position、direction 与显式 null-collision reference 一致。
- fixed seed 下 hit event 序列和 isotope births 不变。

## 退出条件

- [ ] 阶段 A 后 miss 不改变该步 EM/MCS semantics。
- [ ] 中间目标 miss `<0.05%`；最终认证 package miss `=0`。
- [ ] 修复前后 isotope birth expectation 无可测重归一化。
