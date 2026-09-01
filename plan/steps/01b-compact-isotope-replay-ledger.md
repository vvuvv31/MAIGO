# 01B：Compact isotope replay/outcome/transition ledger

## 目标

在不改变 CINEL02 sampler、rate、stopping、MCS 或 straggling 的前提下，
把 secondary replay 的候选、命中、缺失和非法事件按 isotope、target、cascade
reaction_generation 与 50 MeV/u energy bin 分开记录；同时记录 parent handoff 和
generated/queued isotope transition，作为后续 reaction-survival 与 package
auditor 的输入。

## 共享 schema

`Cinel02ReplayLedgerSchema` 定义固定布局：

- replay status：`18 × 2 × 3 × 8 × 5`，顺序为
  `projectile isotope × target(H/O) × reaction_generation(0..2) × energy bin × status`；
- parent outcome：`18 × 2 × 3 × 2`，outcome 为 `continued/killed`，并累加
  incident、parent-after、local、export、import kinetic energy；
- transition：`18 × 18`，分别保存 generated 与 successfully queued 的 count 和
  child kinetic energy。

所有数组同时存在于 device、`TransportResult` 和 JSON，数组长度由 schema 常量
驱动，禁止在 decoder 中另写 magic offset。

## 执行内容

1. primary 和 secondary CINEL02 replay 在 hazard/collision point 记录
   `collision_candidate`；event lookup miss 记录 `replay_lookup_miss`；post-EM energy 低于 cutoff 记录 `post_em_below_cutoff`；越界 product
   或非法 parent status 记录 `replay_invalid_event`；合法 event 记录
   `replay_valid`。
2. secondary parent valid replay 记录 isotope-resolved outcome 与 signed
   handoff energy；CINEL02 role-0 product 分别记录 generated transition，成功写入
   secondary queue 后再记录 queued transition。
3. `accumulate_transport_result()` 合并全部新数组；energy-ledger JSON 输出
   layout、单位、isotope 名称和扁平数组。
4. synthetic accumulator regression 覆盖所有新数组，确保拆分 batch 与单次运行
   结果一致。

## 200 MeV/u G1 100k 证据（2026-09-01）

- Config：`config/beam_200MeVu_cinel02_e200light107_g1_transitiondiag4_100k_xy04.yaml`
- Seed：`2026095100`；histories：`100000`；3D scorer：`200×200×800`，
  `0.4×0.4×0.5 mm`，80×80 mm FOV；本机 RTX 2080Ti / CUDA SYCL。
- Package SHA256：`8a54b8544aea484fa3ff649fc372c22d4b48deff4c2fe37dbd31b2f7f6adb25d`
- Rate SHA256：`aa811ff18684a6a8f81f79fa38f1130b4103555ea2932abb77f7163e1c160ed7`
- Config SHA256：`c29cbe1a4957035294e4fdd1abc3798740cdf723b8aa0b2dc3cc7281e4cf9299`
- Output：`out/beam_200MeVu_cinel02_e200light107_g1_transitiondiag4_100k_xy04/energy_ledger.json`
- energy balance：`0.0153629914825`；沙盒外 2/2 CTest 通过。

Replay status 总计（旧四状态基线）：candidate `71524`、valid `71318`、no-event `206`、invalid
`0`。01B.1 将 no-event 拆为 lookup miss 与 post-EM cutoff，并保留 candidate = valid + lookup_miss + invalid + cutoff。secondary generation-1
no-event 为 `206/33512 = 0.615%`。按 isotope 的 no-event 比例最高的是
`2H 1.779%`、`3H 0.798%`、`7Be 0.490%`、`6Li 0.358%`；`9Be` 和 `10Be` 本轮
均为 0/31 与 0/49。所有 reaction-import energy 仍为 0。

species transport ledger 的 signed closure 继续显示：light-ion p/d/4He 仍分别
约 `+31.57%/+17.30%/+17.12%`，而显式重离子通常 `<0.05%`。因此 01B 只完成了
可定位性建设，未将 residual 误判为 MCS/FOV 或 sampler 问题；下一步进入
deterministic package auditor 与 replay-support consistency。

## 退出条件

- [x] primary/secondary isotope-resolved replay status 已写入 JSON。
- [x] parent outcome 及 generated/queued transition 已写入 JSON。
- [x] 新数组支持多批次 accumulator，synthetic regression 覆盖通过。
- [x] 本机 RTX 2080Ti 的 200 MeV/u G1 100k 运行完成，status partition 守恒。
- [ ] light-ion residual 的 physics 根因尚未解决；留给后续 reaction-survival
  / stopping optical-depth 步骤，不在 01B 修改 sampler/yield。
