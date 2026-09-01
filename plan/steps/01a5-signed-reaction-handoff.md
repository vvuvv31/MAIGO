# 01A.5：Signed secondary reaction handoff closure

## 目标

验证每条次级 CINEL02 parent track 在 reaction replay 前后的能量交接是否存在未记账的 kinetic import。此步骤只增加诊断，不改变 event sampler、核反应概率、stopping、MCS 或 straggling。

## 记账定义

对每个有效 replay，令 `E_before` 为 post-EM collision state 的 parent kinetic energy，`E_parent_after` 为 package 的 surviving-parent kinetic energy（killed parent 为 0），`E_local` 为 process-local deposit：

```text
delta_handoff = E_before - E_parent_after - E_local
reaction_export = max(delta_handoff, 0)
reaction_import = max(-delta_handoff, 0)
```

species ledger 的闭合式为：

```text
queued_birth_kinetic + reaction_import_kinetic
  = continuous_deposit_all + nuclear_local_deposit_all
  + terminal_deposit_all + boundary_escape_kinetic
  + reaction_export_kinetic + step_limit_escape_kinetic
```

`reaction_import_kinetic` 是第 11 个 metric，所有 metric 保持非负；因此可以安全合并多批次。

## replay δE 诊断

对每个有效 package replay、按 projectile isotope 累计：

- `delta = package incident energy (MeV/u) - runtime incident energy (MeV/u)`；
- `Σdelta`、`Σ|delta|`；
- delta positive/negative/valid counts。

JSON 中明确单位和符号，数组使用与 species ledger 相同的 18 isotope 顺序。该量用于识别 window sampling 与 runtime collision state 的能量偏差，不直接作为 physics correction。

## 执行记录

1. 更新 host/device 共享 schema 与 device atomic buffers。
2. primary/secondary 有效 replay 都记录 incident δE；secondary parent 同时记 import/export。
3. 更新 `TransportResult`、多批次 accumulator 和 energy-ledger JSON。
4. 用 synthetic positive/negative handoff fixture 检查数学闭合和 accumulator composability。
5. 在本机 RTX 2080Ti 上跑 200 MeV/u、G1、100k、seed `2026095100`，比较 p/d/t/He4 与重离子的 closure。

## 验收/退出条件

- [x] reaction import/export 使用互斥的 signed split。
- [x] replay δE 按 isotope 输出，且 positive/negative/valid counts 可合并。
- [x] `carbon_tests` 与 `inelastic_package_v2_tests` 在沙盒外 SYCL runtime 下通过。
- [x] p/d/He4/t 的 import-aware closure 已测量；本轮 import=0，残差已明确不由 kinetic handoff 解释，并转交 01B/后续 reaction semantics 定位。
- [x] 200 MeV/u 100k 的诊断结果写入输出目录，并记录 package/config/rate hash。

## 当前结果（2026-09-01）

- 编译成功；沙盒外 `ctest`：2/2 通过。
- 200 MeV/u G1 100k 运行成功，energy balance `0.01536299`。
- replay import 在本轮各 isotope 均为 0；p/d/He4 closure 仍约 `+31.57%/+17.30%/+17.12%`，因此 kinetic handoff import 假设被证伪，不能解释 light-ion residual。
- `ΣδE/u` 已输出；Step 01B 已完成 isotope replay/outcome/transition 分层，下一步进入 deterministic package auditor 与 replay-support consistency。
