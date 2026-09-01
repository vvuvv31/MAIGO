# 04：MCS-only Species/FOV 标定

## 目标

在 yield、birth angle/energy 与 range 可信后，独立验证 species-aware MCS 和 finite-FOV acceptance，不用 full-cascade integrated dose 拟合 MCS。

## 运行矩阵

- Ions：C12、Li6/7、Be6/7/9/10、B10/11；需要时加 p/d/t、He3/4/6。
- Energy：25/50/75/100/150/200 MeV/u。
- Physics：nuclear off，首轮 straggling off，仅 stopping+MCS。
- Scorer：宽横向 phantom 与正式 80×80 mm FOV 同时输出 3D dose。
- Range fractions：0.1/0.25/0.5/0.75/0.9 记录 angular RMS、lateral sigma、r90/r95、FOV survival。

## 阶段 A：Step-size convergence

1. secondary max step 运行 `1× / 0.5× / 0.25×`。
2. 检查当前 endpoint angular kick 对 sigma/FOV fraction 的步长依赖。
3. 若减半 step 后 lateral sigma 或 FOV fraction 变化 >0.3%，先实现 random-hinge 或 Fermi–Eyges correlated lateral displacement。

## 阶段 B：2GR 物理修复

1. 先采样完整 core Gaussian、tail Gaussian 和 power-law tail。
2. 对最终 angle 统一施加 species factor，不只修 `sigma_c`。
3. 第一版：`theta_final = theta_2GR × R_species(Z,A,E,t) × f_residual(group,E,range_fraction)`。
4. `R_species` 优先使用同 E/step 的 `Highland(Z,A)/Highland(reference)` 比值；reference species 必须逐位不变。
5. `f_residual` 仅允许 H/He/Li-Be/B-C 接近 1 的 LUT，不承担 isotope physics。

## 当前假设与反证

- Be6/7 当前可能过于前向，导致 FOV dose 高；Li7 可能过度散射，导致 FOV dose 低。
- Be9/10 正确 species scaling 可能使散射更弱、FOV dose 更高；不能期待 MCS 修复它们的全部正偏差。

## 退出条件

- [ ] mono-ion lateral sigma `<1%`，r90/r95 `<2%`。
- [ ] finite-FOV deposited fraction `<1%`。
- [ ] 宽 FOV 下 MCS on/off integrated deposition 变化 `<0.2%`。
- [ ] 减半 step 后 sigma/FOV fraction 变化 `≤0.3%`。
- [ ] 不使用 species full-cascade dose 作为拟合目标。
