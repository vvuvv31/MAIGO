# 05：Be/Li Stopping 与 Range Regression

## 目标

证明显式 isotope stopping table 和 range-energy transport 正确，将 unrestricted transport deposition 与 FOV/MCS 分离。

## 运行矩阵

- Ions：Be6/7/9/10、Li6/7，回归加 C12、B10/11。
- Energy：25/50/75/100/150/200 MeV/u。
- Physics：nuclear off、MCS off、straggling off。
- Geometry：宽 FOV，防止横向逃逸；只用 3D scorer 横向求和。

## 记录

1. CSDA/mean range、range-energy curve、depth dE/dx。
2. stop 前 residual KE、cutoff deposit、unrestricted total deposited KE。
3. stopping table species index/hash、missing/fallback counter。
4. max step `1× / 0.5× / 0.25×` convergence。

## 永久约束

- 仅允许显式 ion stopping table。
- 禁止 Z²/36 scaling、C12 fallback、same-element isotope alias 和 global stopping scale。
- Be6 独立表必须有单元测试；缺表必须 fail-fast。

## 退出条件

- [ ] mean range `<0.5%`，目标 `<0.3%`。
- [ ] 宽 FOV integrated deposited energy `<0.2%`。
- [ ] missing/fallback/alias counter `=0`。
- [ ] max step 减半后 range/species dose 变化 `<0.2%`。
- [ ] 若 `D_all/E_birth` 通过但 `D_FOV/D_all` 失败，问题明确转入 Step 04，不改 stopping。
