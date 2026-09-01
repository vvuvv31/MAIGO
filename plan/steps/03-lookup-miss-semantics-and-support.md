# 03：Lookup Miss Semantics 与 Event Support

## 目标

先保证 replay miss 不扭曲 EM/MCS transport，再使 rate coverage 与 event support 严格一致。

## 问题

rate group 可在连续能区插值为非零，但 event replay 要求当前 `±0.51 MeV/u` 真正有 event。200 MeV/u G1 compatibility baseline 实测 `192/71524 = 0.268441%` lookup misses；另有 14 个 post-EM cutoff，不能混作 lookup miss。

## 阶段 A：Null-collision transport

1. [x] miss 保留已走路径的 continuous EM loss 和 post-EM state。
2. [x] miss 也必须执行与普通步一致的 MCS/lateral displacement，不得人工直线飞行。
3. [x] 不产生 products、local nuclear deposit 或 parent final-state 变化。
4. [x] 记录 `(Z,A,target,E,reaction_generation)` miss histogram 和 associated KE；当前 compact ledger 不含 depth，spatial discrepancy 留到后续 optical-depth 阶段再加。

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

### 03B-2A：occupancy-aware support audit（2026-09-01）

- [x] 只读使用 rate/package census 与 10-MeV/u replay ledger，按 isotope×target×reaction_generation×energy cell 汇总 candidate、valid、miss、cutoff 及 rate/replay energy。
- [x] 将 miss 按 occupied-cell mean provisional 分类为 `raw_or_compiler_support_gap`、`post_em_support_boundary`、`index_or_lookup_anomaly` 或 `rate_occupancy_or_interpolation_gap`；不修改 runtime rate、CDF、tolerance 或 target mix。
- [x] 6Be 从 audit 中排除并标记 `non_transportable_prompt_decay`，不再作为 missing rate coverage。
- [x] 输出：`startup/package_tools/audit_cinel02_replay_support_occupancy.py`、`plan/artifacts/cinel02-replay-support-occupancy-e200-g1-topascompat/audit.{json,csv}`。
- [x] canonical compatibility 100k 结果：transportable isotope candidate=71524、lookup miss=192（0.268441%）；Li6/Li7、Be7/Be9/Be10 miss 均为 0。174/192 miss 落入 raw/compiler support-gap provisional bucket，18/192 为 index/lookup anomaly provisional bucket。
- [ ] 由于 ledger 没有 per-collision path length，本步只报告 `-log(1-miss fraction)` optical-depth proxy，不能宣称物理 integrated optical depth 已闭合。


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

### 03B-2B：bounded source/compiler consistency（已完成，仅 transportable isotopes）

本阶段是 correctness cleanup，不是 Be/Li 剂量收敛步骤。只处理 03B-2A 实际占用的
174 个 provisional source/compiler gap 与 18 个 lookup/index anomaly；不改变 rate、target
CDF、tolerance 或 isotope yield。

1. [x] raw/package provenance census 完成：34 个 source-backed groups 全部匹配，2 个 6Be
   groups 按 non-transportable policy 缺失；compiler/package 丢 support 为 0。
2. [x] 14,293,976 个 exact node/window replay probes 全部通过；18 个 anomaly 没有确认的
   index/lookup bug，保留为 provisional runtime sparse-support 分类。
3. [x] 没有 raw-present/compiler-dropped 项，因此不修改 source/compiler/package manifest，
   不静默改变 H/O target mix，也不新增 suppressed-hazard physics。
4. [x] 不跨大能差寻找 nearest event，不扩大 tolerance 来隐藏 gap；03B-2A 已证明 Be/Li miss=0。
5. [x] 不在 E_rate 上直接乘 runtime support mask；没有 occupancy 证据要求 support-boundary
   segmentation。192 个 miss 作为 correctness residual 携带。

bounded 检查已完成，立即进入 reaction-survival × stopping-residence 主线；不得以 dose
改善作为本阶段验收。

## 测试

- [x] 合成 gap 的 MCS 判定：普通步、rate-covered/event-uncovered null collision、
  invalid replay、valid replay、cutoff 和禁用 MCS。
- [x] miss 步保留 post-EM state，并通过 GPU 回归确认 status partition 与有效 replay
  统计不变。
- [x] 03B-0 census 的区间 union/zero-rate 过滤 synthetic tests。
- [x] 03B-1 raw campaign/compiler root-cause 与 03B-1R compatibility policy 判定完成；不补充 Be-6 data。
- [x] 03B-2A occupancy 分类与 Be6 non-transportable policy test 已补充。
- [x] 03B-2B Python regression 覆盖 raw/compiler/package provenance key 表、exact-energy
  node/window lookup 和 `(Z,A)` mismatch；生产 census 覆盖 H/O 与 0/1/2 `reaction_generation`，
  不预先实现 runtime support mask。

## 退出条件

### 阶段 A

- [x] lookup miss/invalid 不再跳过该步正常 secondary MCS。
- [x] valid replay 不重复施加普通步 MCS；post-EM cutoff 不施加超出 cutoff 的 MCS。
- [x] 沙盒外编译、CTest 和 200 MeV/u 100k GPU 回归完成。

### 阶段 B（03B-2B 已完成）

- [x] 03B-2A 已按实际 occupancy 分类 miss；不把 0.268% global miss 误报为 Be/Li dose 根因。
- [x] 174 个 provisional gap（20 cells）与 18 个 provisional anomaly（9 cells）已有
  source/package provenance；34 个 source-backed groups 无 raw/package mismatch，6Be 两组按
  policy 排除。
- [x] 14,293,976 个 exact-energy node/window probes 全部通过；未确认 index/lookup bug，
  transportable Be/Li miss 仍为 0，其余 192 个 miss 显式携带为 runtime sparse-support residual。
- [x] 没有 source/compiler 修复项；未扩大 tolerance、使用 nearest-event fallback，未在
  E_rate 上静默乘 runtime mask，也未改变 isotope birth expectation。
