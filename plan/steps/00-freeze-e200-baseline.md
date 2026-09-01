# 00：冻结 200 MeV/u 可复现基线

## 目标

将开发用的 `c2e25b8` TOPAS-compatibility profile 与唯一的 package、TOPAS reference、geometry/FOV 和 scorer contract 绑定。历史 `becf880` 仍保留作 native-transport 对照；后续诊断可增加，但禁止改变核反应概率、event selection 或 isotope yield。

## 固定内容

1. 代码：commit `c2e25b832fbc015c611eb4d233da4a164abefd23`，本机 RTX 2080Ti/sm_75。
2. GPU config：`config/beam_200MeVu_cinel02_e200light107_g1_topascompat_100k_xy04.yaml`，显式 `cinel02_topas_compatibility_mode: true`；200 MeV/u、100k、seed `2026095100`、G1、3D scorer。
3. Package：`research_hybrid_e200light107.cinpkg`，记录 package/manifest/rate SHA256。
4. TOPAS：100k charged-origin 和 isotope-origin reference；1M full-cascade job 485。
5. Geometry：200×200×800，0.4×0.4×0.5 mm，80×80 mm FOV；IDD 仅从 3D scorer 横向求和。
6. 保存 canonical config、seed、histories、TOPAS job ID、package hash、scorer header 和 artifact 路径。

## 历史 native-transport 对照

`becf880` 的旧结果：Primary C -0.492%，Secondary C -1.452%，B +0.963%，Be +4.844%，Li -3.668%，He +0.767%，Z1 +1.751%，Total -0.511%。该表不再作为 compatibility profile 的开发基线。

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


## c2e25b8 TOPAS-compatibility canonical baseline（2026-09-01）

- Manifest：`plan/baseline-e200-c2e25b8-topascompat.json`。
- GPU ledger/quality：`out/beam_200MeVu_cinel02_e200light107_g1_topascompat_100k_xy04/{energy_ledger,quality_report}.json`。
- 3D species analyzer：`plan/artifacts/species-idd-100k-e200-g1-topascompat/analysis.json`。
- Compatibility A/B 已确认：Be6 produced=281 不变，B queued=0、discarded kinetic=130184.242188 MeV；nuclear interaction count 不变。
- 当前描述性 species integral（TOPAS total_species 100k 与 GPU compatibility 100k，独立 seed，不是 CI）：Primary C +7.278%，Secondary C +3.327%，B +6.539%，Be -2.365%，Li +0.976%，He +5.183%，Z1 +4.207%，charged total -0.569%。
- Be6 gate 暂不使用 relative percent：TOPAS 的 `ChargedOriginDoseToMedium` Be6 scorer 为 0.236187 Gy，而 compatibility GPU depositing-track `species_be6` 为 0；这是 origin-attribution 与 depositing-isotope 定义不一致，必须先修 scorer contract。
- accounting relative residual 为 1.5363886%，physical relative residual 为 1.5906321%；前者吸收显式 TOPAS compatibility sink，后者保留 reference sink 后的物理缺口，均不是本步骤的物理收敛声明。

## 退出条件调整

- [x] compatibility config 显式固定 `cinel02_topas_compatibility_mode: true`，provenance/hash 已入 manifest。
- [x] 200 MeV/u 100k GPU compatibility baseline 已重跑并记录双 closure。
- [ ] Be6 relative-dose gate：等待匹配的 depositing-track isotope scorer；在此之前只验 production count/KE 与 absolute near-zero compatibility sink。
