# 09：Stopping Power、Straggling、MCS 与 Lateral Halo

## 目标

在birth physics固定后验证17种isotope的EM输运和横向模型，禁止用输运参数补偿错误composition。

## 实施步骤

1. 17种isotope在10--400 MeV/u运行mono-ion 3D dose。
2. 比较range、积分、peak、straggling、core sigma；确认per-species LUT均加载。
3. 禁止Z²/36 fallback，缺表直接失败。
4. 分别验证primary-only core、mono-fragment core、full-physics halo。
5. composition修复后重做double Gaussian：每1 mm、halo≥1.8 core、weight≥1%、改善≥0.8%。
6. 仅当mono-ion同方向偏差时引入energy/species-dependent MCS，不先改全局1.40。
7. relative-error图明确dose threshold；截断图不标为unclipped。

## 验收

- [ ] 17 isotope range/integral ≤1%，IDD NRMSE ≤2%。
- [ ] core sigma中位差≤2%，halo sigma中位差≤5%。
- [ ] 100k稀疏halo报告CI和有效点数，不据此调参。
