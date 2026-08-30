# 步骤 02：建立完整事件能量守恒

## 目标

消除 target KE 的“免费能量”、负 residual 仍被接受以及逐碎片 clip 导致的谱偏置。一次事件要么完整守恒并被接受，要么整体重采样。

## 实施任务

1. 定义统一 ledger，至少包含 incident、projectile charged、target charged、neutron、remnant kinetic、excitation、local deposit、escaped neutral、numerical residual。
2. 把所有 projectile/target fragment、neutron 和 remnant KE 纳入同一预算。
3. paper-minimal 初始模式使用 `Q=0`；Q/excitation 的扩展实现留到步骤 07。
4. 删除 `fragment_energy = min(fragment_energy, remaining_room)` 一类逐碎片截断。
5. 任意产品令总能量超过 available energy 时，拒绝整个事件并重采样；设置有上限的 retry，超限必须计数并安全失败。
6. 明确定义 escaped neutral 与 local deposit，任何能量只能属于一个 ledger 项。

## 守恒式

```text
incident + Q
= projectile_charged + target_charged + neutron
 + remnant_kinetic + excitation + local_deposit
 + escaped_neutral + numerical_residual
```

## 测试

- 单事件构造：恰好守恒、轻微超预算、target KE 超预算、neutron 超预算、retry exhaustion。
- 至少百万级事件统计 signed residual 分布和拒绝率，按 beam energy、H/O target 分组。
- 比较修复前后 100–400 MeV/u 100k/1M smoke runs，禁止用 normalization 修正。

## 硬门槛

```text
negative residual accepted count       = 0
|numerical residual| / incident energy < 1e-5
double-counted ledger entries          = 0
non-finite ledger values               = 0
```

## 建议提交

```text
fix(fred): enforce complete signed event energy closure
```

