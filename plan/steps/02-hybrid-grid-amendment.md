# 02 补充：Hybrid-log 主资格网格

本文件覆盖 `02-package-coverage-auditor.md` 中“固定 1 MeV/u 主矩阵”的规定。
原 1 MeV/u 报告只作为详细诊断基线，不再作为主资格矩阵。

## 权威网格

1. 0--10 MeV/u：1 MeV/u。
2. 10--50 MeV/u：2 MeV/u。
3. 50 MeV/u 以上：相邻边界比 1.05 的几何网格。
4. runtime 连续能量 lookup 与 ±0.51 MeV/u tolerance 不变。
5. 宽 bin 内 rate 变化超过 10% 时不得用合并统计直接认证，须拆分或使用子 bin 加权。

## Runtime demand buffer

- 维度：17 species × H/O × G0/G1/G2 × hybrid energy bin。
- 每 cell 至少记录 query、hit、miss、candidate-count sum/min。
- 400 MeV/u 内为 73 energy bins；四个 uint64 counter 约 238,272 bytes。
- G1 demand-weighted coverage ≥99.99%，G2 ≥99.9%。

## 当前结果

- 17-species 主矩阵：2,524 occupied cells。
- 650 qualified，1,874 unqualified。
- 1,441 cells 未通过 event-count/ESS。
- 1,452 cells 的内部 rate variation 超过 10%。
- 证据：`plan/artifacts/cinel02-package-coverage-e400/coverage_hybrid_runtime17.json`。

步骤 02 尚未完成；下一项为实现动态 hybrid demand buffer。
