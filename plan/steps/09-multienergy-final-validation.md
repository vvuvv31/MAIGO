# 09：100/200/300 MeV/u 最终验收

## 目标

先用严格 200 MeV/u physics gate 收口，然后保持所有 physics 参数不变推广到 100/300 MeV/u。400 MeV/u 仅为非阻塞外推。

## 统计阶梯

1. 100k smoke：功能、closure、方向、显著回归。
2. 1M physics gate：species/isotope 主结论。
3. 必要时 10M：低 yield isotope/channel。
4. 最终至少 5 个独立 seed，报告 mean、95% CI 和 effective counts。

## 固定 scorer

- 无过滤 total、charged-origin 7 species、Be/Li isotope、other charged、neutral-origin、unclassified。
- 只用 3D scorer 横向求和；正式 FOV 与宽 FOV acceptance scorer 同时运行。
- 绘图深度限制为 TOPAS total Bragg peak 的 1.2 倍，同时保留 unrestricted/full-depth ledger。

## 开发目标

- Total mean `±1%`。
- Secondary C/B/Be/Li/He/Z1 mean `±1.5%`，为最终 2% 留 0.5% 统计缓冲。
- 低统计 isotope 用 deterministic expectation + CI，不强迫单 seed ratio 稳定到 2%。

## 最终硬门槛

- [ ] Total：积分/峰值 `≤2%`，峰位 `≤0.5 mm`，IDD NRMSE `≤1%`。
- [ ] 七类 species：积分 `≤2%`，峰位 `≤1 mm`，IDD NRMSE `≤2%`。
- [ ] mono-ion MCS/range/straggling 各自通过 Steps 04–06。
- [ ] hierarchical closure、lookup support 和 package qualification 通过。
- [ ] 100/200/300 MeV/u 使用同一套 physics 参数，禁止 per-energy scale。
- [ ] 确定 G1/G2 production default 并记录理由。

## 失败回滚

按最早失败层回退：birth K → unrestricted transport `f_dep` → FOV `f_FOV` → cascade survival → package support。禁止末端经验拟合。
