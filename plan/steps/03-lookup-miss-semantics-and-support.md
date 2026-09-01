# 03：Lookup Miss Semantics 与 Event Support

## 目标

先保证 replay miss 不扭曲 EM/MCS transport，再使 rate coverage 与 event support 严格一致。

## 问题

rate group 可在连续能区插值为非零，但 event replay 要求当前 `±0.51 MeV/u` 真正有 event。200 MeV/u G1 已观测 `192/33512 = 0.573%` secondary misses。

## 阶段 A：Null-collision transport

1. [x] miss 保留已走路径的 continuous EM loss 和 post-EM state。
2. [x] miss 也必须执行与普通步一致的 MCS/lateral displacement，不得人工直线飞行。
3. [x] 不产生 products、local nuclear deposit 或 parent final-state 变化。
4. [x] 记录 `(Z,A,target,E,generation,depth)` miss histogram 和 associated path/KE。

### 阶段 A 实施与验证总结

- 代码提交：`eb80a89 fix(cinel02): preserve secondary transport on replay miss`。
- 核心实现：新增 `secondary_replay_succeeded` 状态；只有有效 CINEL02 event
  才抑制当前步的普通 secondary MCS；lookup miss、invalid event 和 post-EM
  cutoff 继续使用 post-EM 能量/位置，且 miss/invalid 在能量高于 cutoff 时执行
  正常 MCS。
- 回归测试：新增 `cinel02_should_apply_secondary_mcs` synthetic regression，覆盖
  普通步、miss/invalid null collision、有效 replay、cutoff 和禁用 MCS；沙盒外
  `cmake --build build -j2` 成功，CTest `2/2` 通过。
- GPU 证据：本机 RTX 2080 Ti/sm_75，CUDA SYCL，200 MeV/u、G1、100,000
  histories、seed `2026095100`，3D scorer 200×200×800（0.4×0.4×0.5 mm，
  80×80 mm FOV）。旧输出：
  `out/beam_200MeVu_cinel02_e200light107_g1_transitiondiag4_100k_xy04/energy_ledger.json`；
  新输出：
  `out/beam_200MeVu_cinel02_e200light107_g1_step03_nullmcs_100k_xy04/energy_ledger.json`。
- 结果：replay status 完全不变，`candidate/valid/lookup_miss/invalid/cutoff`
  均为 `71524/71318/192/0/14`；`nuclear_interactions=71524`。generated/queued
  transition 总数由 `379379` 变为 `379390`（约 `0.003%` 的路径重分支），说明
  null-collision MCS 会改变极少数后续轨迹，但没有改变 replay status 或 package
  支持统计。总沉积能量由 `228097443.355` 变为 `228097665.701` MeV，差
  `222.346` MeV（约 `9.7e-7`）；未跟踪能量减少 `475.848` MeV。全局 quality
  ledger 的既有残差仍约为 `3.687e6` MeV（相对 `1.536%`，status=`non_production`），
  本次仅改变约 `2.15e2` MeV，未引入新的 queue overflow；因此不能把全局残差称为
  已闭合，后续仍需按 species/reaction ledger 处理。
- 本阶段结论：阶段 A 的 transport semantics 已修正并通过验证；这不是 physics
  yield/package 修改，p/d/He4 的既有 residual 仍未解决。

## 阶段 B：Support-aware rate

### 03B-0：18 isotope coverage census

- [x] 只读扫描 rate CSV 与 CINPKG03 global energy nodes。
- [x] 确认 6Be rate group 存在但 H/O 全为零，且 package 无 6Be event-support nodes。
- [x] 确认 6Li/7Li、7Be/9Be/10Be 等其他关键 isotope 具有正 rate 与 package nodes。
- [x] 输出 `plan/artifacts/cinel02-rate-package-census-e200-g1/`；详见
  [03B-0 record](03b0-rate-package-coverage-census.md)。

### 03B-1：Be-6 coverage root-cause gate（已完成）

1. [x] 追溯 raw exposure/contract/summary，确认缺少 6Be projectile campaign，而非 compiler 过滤丢失。
2. [x] 确认 Be-6 仅作为 C-12 direct child 生成（raw 计数 27,186），没有 Be-6 projectile interaction。
3. [x] 保持 isotope alias 禁止；独立 6Be 数据补充前不静默填 rate/event。
4. [x] 03B-1R：按 Job 495 reference semantics 实现 TopasCompatKill；不补充独立 6Be+H1/O16 exposure/package。

### 03B-2：support-aware rate consistency（待执行，仅 transportable isotopes）

1. compiler/manifest 为每个 rate group 输出真正 event-support intervals/mask。
2. runtime 仅在 tolerance window 内存在 event 时标记 replay-covered。
3. 不跨大能差寻找 nearest event，不扩大 tolerance 来隐藏 gap。
4. 检查 support mask 对 rate/reaction CDF 的影响，不允许无声重归一化 isotope yield。

## 测试

- [x] 合成 gap 的 MCS 判定：普通步、rate-covered/event-uncovered null collision、
  invalid replay、valid replay、cutoff 和禁用 MCS。
- [x] miss 步保留 post-EM state，并通过 GPU 回归确认 status partition 与有效 replay
  统计不变。
- [x] 03B-0 census 的区间 union/zero-rate 过滤 synthetic tests。
- [x] 03B-1 raw campaign/compiler root-cause 与 03B-1R compatibility policy 判定完成；不补充 Be-6 data。
- [ ] 03B-2 仍需补充 rate-covered/event-uncovered、boundary hit、H/O 切换、G1/G2
  的 support-aware synthetic tests，以及逐 collision 的 support mask 验证。

## 退出条件

### 阶段 A

- [x] lookup miss/invalid 不再跳过该步正常 secondary MCS。
- [x] valid replay 不重复施加普通步 MCS；post-EM cutoff 不施加超出 cutoff 的 MCS。
- [x] 沙盒外编译、CTest 和 200 MeV/u 100k GPU 回归完成。

### 阶段 B（Step 03 尚未完成）

- [ ] 中间目标 miss `<0.05%`；最终认证 package miss `=0`。
- [ ] 修复前后 isotope birth expectation 无可测重归一化。
- [ ] compiler/manifest 输出 event-support intervals/mask，runtime 与 replay support
  严格一致；不得扩大 tolerance 或使用 nearest-event fallback。
