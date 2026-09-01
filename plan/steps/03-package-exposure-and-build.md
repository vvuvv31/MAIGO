# 03：Projectile/Energy Exposure 与 Package Family 重建

## 目标

覆盖17种带电同位素、H/O target、0--400 MeV/u，并达到步骤02的统计门槛。

## 同位素域

p/d/t；He3/He4/He6；Li6/Li7；Be7/Be9/Be10；B8/B10/B11；C10/C11/C12。

## 实施步骤

1. C12 分别运行100/200/300/400 MeV/u厚水 campaign，作为临床路径验证样本。
2. 每种 projectile 做均匀 energy exposure，覆盖0.5--399.5 MeV/u，间隔1 MeV/u；H1/O16分开记录。
3. 小批量测 reaction efficiency，再计算达到统计门槛所需 histories。
4. TOPAS 总资源≤192 CPU/128G，sbatch串行/分组；InvalidAccount等待1--3分钟复查。
5. raw capture 做 CRC、collision/final state、local deposit、unsupported ledger审计。
6. 按 projectile 分 shard 编译；G1只加载C shard，G2按实际需求加载。
7. auditor通过后才设 `statistically_qualified=true`。

## 验收

- [ ] 17 isotope × H/O 需求域无未解释 gap。
- [ ] unsupported product energy fraction ≤0.2%。
- [ ] package family满足RTX 2080Ti显存预算。
- [ ] 独立validation campaign落在bootstrap 95% CI内。
