# 00：冻结 200 MeV/u 可复现基线

## 目标

将 `becf880` 与唯一的 package、TOPAS reference、geometry/FOV 和 scorer contract 绑定。后续诊断可增加，但在 TOPAS job 485 和 deterministic auditor 给出一致结论前，禁止改变核反应概率、event selection 或 isotope yield。

## 固定内容

1. 代码：commit `becf880`，本机 RTX 2080Ti/sm_75。
2. GPU config：200 MeV/u、100k、seed `2026095100`、G1、3D scorer。
3. Package：`research_hybrid_e200light107.cinpkg`，记录 package/manifest/rate SHA256。
4. TOPAS：100k charged-origin 和 isotope-origin reference；1M full-cascade job 485。
5. Geometry：200×200×800，0.4×0.4×0.5 mm，80×80 mm FOV；IDD 仅从 3D scorer 横向求和。
6. 保存 canonical config、seed、histories、TOPAS job ID、package hash、scorer header 和 artifact 路径。

## 回归基线

Primary C -0.492%，Secondary C -1.452%，B +0.963%，Be +4.844%，Li -3.668%，He +0.767%，Z1 +1.751%，Total -0.511%。

## 退出条件

- [x] provenance manifest：`plan/baseline-e200-becf880.json`。
- [x] 只诊断构建与基线 IDD 逐项一致。
- [x] job 485 输入、seed、UUID、资源、hash 和输出路径已入档。

## 完成证据（2026-09-01）

- TOPAS job 485 完成 1,000,000 histories：737,255 interactions、5,530,786 products。
- C12+O16、50--100 MeV/u Be9：TOPAS `547/46962 = 0.0116477`，package `2982/242173 = 0.0123135`，差异约 1.2σ。
- 100k 的 +33.8% 表观差主要是统计波动；sampler/package yield 继续冻结。

## 禁止项

- 不提交 raw/3D dose artifact；不使用 dose/yield/MCS 经验 scale。
- job 485 和 deterministic auditor 收口前不改 sampler/package yield。
