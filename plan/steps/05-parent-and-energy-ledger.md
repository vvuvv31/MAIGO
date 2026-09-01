# 05：Parent Continuation、Local Deposit 与逐代账本

## 目标

消除 pre/post-EM state混用、parent重复输运、local deposit重复计分和generation-limit能量丢失。

## 固定执行顺序

1. 完成当前step EM loss并只记录一次。
2. 形成唯一post-EM collision state，用它查询rate、target和event。
3. 完成步骤04 reconciliation。
4. local nuclear deposit只记录一次。
5. parent continuation更新原track，不得同时作为secondary。
6. killed parent终止，所有剩余量进入明确ledger。
7. products按真实Z/A、energy、direction、generation入队。

## 账本

按 generation/projectile/target/energy/depth 输出 input、EM、parent-out、charged products、neutral、local、unsupported、cutoff、escaped、overflow、generation-limit energy。

## 验收

- [ ] 单反应测试覆盖 killed/continued parent、neutral、unsupported、overflow、generation limit。
- [ ] post-EM energy是rate和event lookup唯一输入。
- [ ] G0/G1/G2每层相对 residual ≤0.5%。
- [ ] Primary C不再因continuation/local deposit重复而随能量系统上升。

## 2026-09-01 runtime review 结论

- CINEL02/Geant4 `parent_status==0` 表示 `fAlive`，当前 primary 与 secondary 路径均将原 track 更新为 `event.parent_energy_MeV` 和 parent direction。
- `parent_status==2` 表示 `fStopAndKill`；package 加载校验要求其 `parent_energy_MeV <= 1e-4 MeV`，runtime 将 track 置零。因此 code review 中“status 2 幸存但未使用 parent final energy”的风险不成立。
- secondary 的 `sec_step_mm` 在核反应时被截断到 collision distance，`dE` 基于该步长计算；event lookup 使用唯一的 post-EM `sec_e/A`，时序已与 AlongStep 后 PostStep 一致。
- 现有 8-slot ledger 只是全局聚合核反应账本，尚未满足本步要求的 generation/projectile/target/energy/depth 分层 closure；本步仍不标记完成。
- 当前硬验收范围调整为 100--300 MeV/u：先在 200 MeV/u 完成分层 ledger，再以 100/300 MeV/u 检查泛化。400 MeV/u ledger 仅保留为非阻塞外推证据，不启动新的高能任务。
