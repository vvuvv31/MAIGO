# 07：Full-cascade Species Closure

## 目标

在 ledger、lookup、MCS、stopping 和 straggling 收口后，重跑 200 MeV/u G0/G1/G2，将 Li 和 Be isotope 剩余误差归因到 birth、cascade survival 或 transport/FOV。

## 执行

1. 对每个 isotope 输出 G0/G1/G2、H/O、channel、parent-energy、birth-depth。
2. 计算 `K_birth → secondary nuclear survival → D_all → D_FOV`。
3. 每个 secondary nuclear event 做局部 closure，并按 `(generation,projectile,target,Ebin)` 汇总。
4. 比较 G0→G1、G1→G2 的 destruction/regeneration，不以 total cancellation 判断正确性。

## Li 决策树

- `K_birth` 已低：回到 Be/B/C→Li generation/channel yield。
- `K_birth` 匹配但 `f_dep` 低：检查 stopping、straggling、secondary reaction survival。
- `f_dep` 匹配但 `f_FOV` 低：检查 Li7 MCS/FOV acceptance。

## Be 决策树

- Be6：优先查 `f_FOV`，同时检查不稳定核/后代 origin attribution；不做 isotope dose scale。
- Be9/10：species-aware MCS 可能使 FOV dose 更高；剩余正偏差必须由 `K_birth`、`f_dep` 或 cascade survival 解释。

## 退出条件

- [ ] generation/projectile/target/Ebin local residual `<0.1% K_birth`。
- [ ] Li 和每个 Be isotope 都有唯一首要误差层。
- [ ] G2 若比 G1 任一主要 species 恶化 >1 个百分点，保持 G1 为 production default。
- [ ] 不用 total 接近掩盖 species 相反偏差。
