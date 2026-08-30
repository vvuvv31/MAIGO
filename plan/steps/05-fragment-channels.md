# 步骤 05：替换 nearest-remnant fragment channel sampler

## 目标

删除“一个 leading isotope + 一个 nearest residue”的非守恒近似，保证每个 accepted channel 的 A/Z 精确闭合并恢复合理的 charged multiplicity。

## 第一阶段实现

实现 constrained sequential sampler：从 Table-1 CDF 抽 isotope，仅当其适合剩余 A/Z 时加入；无法精确完成时拒绝整个 event。不得把缺失正电荷静默改写成 neutron。

精确条件：

```text
projectile: sumA + remnantA = 12, sumZ + remnantZ = 6
target H:   sumA + remnantA = 1,  sumZ + remnantZ = 1
target O:   sumA + remnantA = 16, sumZ + remnantZ = 8
```

## 最终 GPU 方案

CPU 离线枚举物理允许的 fragment multisets，通过非负优化或最大熵匹配 isotope marginals、charged/neutron multiplicity 和 channel constraints，生成紧凑 CDF LUT；GPU 每事件只做一次 lookup。必须把它表述为“published marginals 的 constrained approximation”，不能声称恢复了未公开的 Newton joint table。

## 测试

- 覆盖已知失败例：`6He + nearest 6Li`、`7Li + 4He` 等缺电荷情况。
- 生成至少 10^7 个 H/O 薄靶事件。
- 输出每 isotope marginal、charged/neutron multiplicity、retry count、invalid channel 和 queue occupancy。

## 硬门槛

```text
A closure                    100%
Z closure                    100%
invalid accepted channel     0
major isotope marginal error < 2%
retained isotope error       < 5%
product queue overflow       0
```

## 建议提交

```text
fix(fred): replace nearest-remnant projectile sampler
test(fred): require exact A/Z closure
```

