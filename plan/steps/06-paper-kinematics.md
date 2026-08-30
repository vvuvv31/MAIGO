# 步骤 06：按论文重实现 Eq. 12–21 运动学

## 目标

修复动态 reference 截断、mixture 权重不一致、只对 A≥10 使用 correlation，以及 heaviest-first + clip 对能谱造成的顺序偏差。

## 规则

```text
p/d/t:                    projectile 与 target 均用完整 Eq. 12 mixture
其他 projectile fragment: Gaussian component
其他 target fragment:     exponential component
```

1. 为每个 target/isotope 定义固定 95 MeV/u reference domain；不得令 `E95,max` 随当前 beam energy变化。
2. 在实际截断域上离线积分 Gaussian/exponential 权重，预计算 `P_gaussian` LUT。
3. `E = E95 * Eprojectile / 95` 只缩放一次。
4. 所有 projectile-origin fragments 都进入 Eq. 13–16 correlation，不设置 A≥10 例外。
5. 全事件超出能量预算时整体拒绝，不逐 fragment clip，不依赖 heaviest-first 排序。
6. 角度在 0°–180° reference domain 抽样；按论文应用 Eq. 21，并明确 proton/neutron 例外和 target-fragment 规则。

## 薄靶验证

95 MeV/u、H/O target，逐 isotope 输出 E/A、theta、E-theta 2D、mixture component fraction、origin。先复现论文分布的形状和矩，再进行水模体运行。

## 验收

```text
energy-spectrum moments < 5%
angle-spectrum moments  < 5%
mixture LUT vs numeric integration within declared tolerance
accepted event energy/A/Z closure 100%
```

同时验证抽样 support 不再随 beam energy 二次增长，事件结果对产品处理顺序不敏感。

## 建议提交

```text
fix(fred): implement paper Eq12-21 without dynamic truncation
```

