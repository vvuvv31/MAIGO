# CINEL02 Species IDD 分层修复计划

## 唯一执行原则

每个 species/isotope 的误差必须先拆成：

`D_FOV = K_birth × f_dep × f_FOV`

- `K_birth`：birth/yield/kinematics 层。
- `f_dep = D_all/K_birth`：stopping、range、straggling、cutoff 和 cascade survival 层。
- `f_FOV = D_FOV/D_all`：MCS、lateral displacement 和 finite-FOV acceptance 层。

未完成分层定位前，禁止修 sampler/yield，禁止 isotope/global dose scale，禁止用 MCS/halo 补偿 event generator。

## 运行约束

- GPU 仅在本机 RTX 2080Ti/sm_75 运行，不提交远程 GPU job。
- TOPAS 仅在 `wuwei@127.0.0.1` 上通过 `sbatch` 运行，数据位于 `/mnt/sda/wuwei`。
- TOPAS 总资源不超过 192 CPU / 128 GB；短暂 `InvalidAccount` 等待 1–3 分钟再检查。
- 仅用 3D dose scorer 横向求和，禁止 1D scorer。
- 不考虑 FP32/FP64 作为修复。
- 仅允许显式 ion stopping tables；禁止 Z² scaling、C12 fallback、isotope alias 和 stopping/dose normalization。

## 当前冻结基线

- Commit：`becf880`。
- Energy：200 MeV/u，100k histories，seed `2026095100`，G1。
- Package：`research_hybrid_e200light107.cinpkg`，window-event sampling `±0.51 MeV/u`。
- Grid/FOV：200×200×800，0.4×0.4×0.5 mm，80×80 mm。
- Species：Primary C -0.492%，Secondary C -1.452%，B +0.963%，Be +4.844%，Li -3.668%，He +0.767%，Z1 +1.751%，Total -0.511%。
- Be isotope：Be6 +10.388%，Be7 +2.730%，Be9 +13.496%，Be10 +6.881%。
- Be6 birth KE +3.82% 但 dose +10.39%，`dose/birth-KE` +6.33%；GPU G0 mean cosine 0.791，TOPAS 0.735。
- Secondary lookup miss：`192/33512 = 0.573%`。
- TOPAS 1M full-cascade job 485 用于 Be9 统计门禁。

## 优先级

`P0 分层 energy closure → P1 deterministic auditor → P1 lookup semantics/support → P1 MCS/FOV → P2 stopping/range → P2 non-C12 straggling → P3 cascade → package gate → 100/200/300 validation`。

## 进度控制

当前完成度：**1/10（10%）**。

当前下一步：**继续 Step 01：定位 p/d/He4 closure 残差，并增加 generation/target/energy/depth sparse 分层；Be/Li 的正式 FOV acceptance 已证实接近 1。**

| 状态 | 步骤 | 主要产出 |
|---|---|---|
| [x] | [00 冻结 200 MeV/u 基线](steps/00-freeze-e200-baseline.md) | `baseline-e200-becf880.json`；job 485 已解析 |
| [ ] | [01 Species 分层 ledger](steps/01-hierarchical-species-ledger.md) | `K_birth × f_dep × f_FOV` 逐层 closure |
| [ ] | [02 Deterministic package auditor](steps/02-deterministic-package-yield-auditor.md) | GPU actual vs package exact vs TOPAS source |
| [ ] | [03 Lookup miss semantics/support](steps/03-lookup-miss-semantics-and-support.md) | null-collision MCS 和 support-aware rate |
| [ ] | [04 MCS-only species/FOV](steps/04-mcs-only-species-fov.md) | step convergence、species-aware full-2GR、FOV acceptance |
| [ ] | [05 Stopping/range regression](steps/05-stopping-range-regression.md) | Be/Li explicit-table range 与 unrestricted deposition |
| [ ] | [06 Non-C12 straggling](steps/06-nonc12-straggling.md) | mean-preserving species-aware fluctuation |
| [ ] | [07 Full-cascade species closure](steps/07-full-cascade-species-closure.md) | Li/Be G0/G1/G2 birth→survival→transport→FOV |
| [ ] | [08 Package 修改门禁](steps/08-package-change-gate.md) | 仅在 job 485 + auditor 同向时修 construction cause |
| [ ] | [09 100/200/300 最终验收](steps/09-multienergy-final-validation.md) | 同一 physics 参数的多 seed 报告 |

## 提交切分

1. `feat(diag): add stratified secondary energy closure ledger`
2. `fix(cinel02): preserve secondary transport semantics on replay miss`
3. `fix(mcs): apply species-aware scaling to full 2gr distribution`
4. `test(mcs): add mono-ion isotope lateral and range regressions`
5. `feat(straggling): add non-c12 secondary loss fluctuations`
6. `test(cascade): add 200mevu species closure regression`

sampler/package commit 不在预定队列中；必须先通过 Step 08 门禁。

## 更新规则

- README 是唯一进度入口；只执行第一个未完成步骤。
- 只有步骤文件的所有退出条件通过并记录证据时才标记 `[x]`。
- 每次记录 commit、canonical config、seed、histories、TOPAS job ID、package/rate SHA256、FOV/scorer header、输出路径和失败项。
- artifacts 保留在本地/15 TB 数据盘，不将 raw/3D dose 提交到 Git。
- 400 MeV/u 不阻塞当前验收；不用它促使 100–300 MeV/u 添加经验 scale。
