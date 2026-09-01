# 06：Non-C12 Secondary Straggling

## 目标

在 stopping/range 与 MCS/FOV 通过后，为非 C12 secondary 增加 mean-preserving、species-aware energy-loss fluctuation。

## 第一版模型

1. C12 保持现有 packaged fluctuation，固定 seed 回归不变。
2. 其他 isotope 使用 `(Z,A,E,step)` 和 effective charge 计算 analytical variance。
3. fluctuation 必须保持 mean loss，不改 isotope birth energy 或 mean stopping power。
4. 低能/cutoff 处保证 non-negative loss、无 clipping-induced mean bias。

## 验证矩阵

- Li6/7、Be6/7/9/10、B10/11、He3/4/6 的 mono-ion wide-FOV campaign。
- 比较 mean range、range sigma、distal 80–20 width、integrated dose。
- straggling off/on 与 step-size convergence。

## 升级条件

只有 analytical width 无法达到 TOPAS 时，才为 Li/Be/B/He 提取独立 Geant4 fluctuation quantile tables。

## 退出条件

- [ ] mean range `<0.3%`。
- [ ] range width/distal width `<3%`。
- [ ] 宽 FOV integrated dose 变化 `<0.2%`。
- [ ] straggling 只改变 width，不改 mean birth/stopping ledger。
