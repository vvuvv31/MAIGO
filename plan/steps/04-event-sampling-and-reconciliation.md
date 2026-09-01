# 04：Event Sampling 与 Collision-energy Reconciliation

## 目标

移除“最近浮点 energy node≈单事件重放”，恢复统计多样性并在实际碰撞能量下守恒。

## 实施步骤

1. 在同 projectile/target 的 `[E-0.5,E+0.5]` MeV/u 窗口收集全部合格事件。
2. 从完整候选集均匀采样，不先缩减到最近 node。
3. 输出候选数、event ID、reuse count、energy delta 和 rejection reason。
4. 对 parent continuation 和 products 三动量施加共同尺度 `lambda`，求解 actual available energy。
5. 保持 mass、direction、channel、multiplicity；禁止把能差直接加入 local deposit。
6. 无物理解时重采样；达到最大尝试次数后记录 miss，验证模式失败。
7. CPU/GPU 共享索引边界、RNG contract 和 fixed-replay vectors。

## 验收

- [ ] event reuse符合均匀抽样，单event贡献≤cell replay的1%。
- [ ] 单反应 reconciliation 相对 residual ≤1e-5。
- [ ] CPU/GPU固定seed选择相同event。
- [ ] coverage/reconciliation miss均显式计数。

## 2026-09-01 runtime review 结论

- 保留 `±0.51 MeV/u` 容差窗口内的完整 event 均匀抽样；不退回 nearest-single-event replay。
- primary/secondary lookup miss 已分别记入 diagnostic slot 4/18，invalid event 记入 5/19。
- secondary miss 前 track 已推进到 post-EM collision state，因此不会丢失动能或停在原地；但它会放弃已抽中的核反应并跳过该步 MCS，会使反应率偏低。
- `cinel02_strict_match=true` 会在任一 miss/invalid 或 hazard 账本不闭合时使运行失败。未认证 package 可仅用 non-strict 做 coverage 诊断，不得用于最终 IDD 验收。
- 200 MeV/u research107 G1 现有 100k ledger 中 secondary hazards/hits/misses/invalid = `33512/33306/192/0`，miss fraction = `0.573%`；因此必须在 package rebuild 前先固定 runtime 验收语义。
