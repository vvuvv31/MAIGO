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

`P0 Step 01A ledger correctness → P0 Step 01A.5 signed reaction handoff → P0 Step 01B isotope replay/outcome/transition → P0 Step 01B.1 replay semantics cleanup → P1 deterministic auditor → P1 replay-support consistency → P1 reaction-survival + stopping optical depth → P1 causal waterfall/physics fix → P2 non-C12 straggling → P3 MCS shape → P4 100/300 regression`。

## 进度控制

当前完成度：**5/13（约 38%）**；Step 03 阶段 A、03B-0 和 03B-1 root-cause gate 已完成，03B-1R 数据补充与 03B-2 仍进行中。

Step 01A ledger correctness、Step 01A.5 signed handoff、Step 01B compact isotope 诊断仪器和 Step 02 deterministic package auditor 已完成；Step 03 阶段 A 已修复 replay miss 的 null-collision MCS semantics，03B-0 已确定 Be-6 rate/package coverage 缺口，03B-1 已确认根因为 source campaign 未运行 Be-6 projectile，当前继续执行 **Step 03 阶段 B：03B-1R 数据补充与 03B-2 support-aware rate consistency**。Step 01B.1 已完成；p/d/He4 的物理 residual 仍未修复，不能把诊断步骤误记为物理收敛。

| 状态 | 步骤 | 主要产出 |
|---|---|---|
| [x] | [00 冻结 200 MeV/u 基线](steps/00-freeze-e200-baseline.md) | `baseline-e200-becf880.json`；job 485 已解析 |
| [ ] | [01 Species 分层 ledger](steps/01-hierarchical-species-ledger.md) | `K_birth × f_dep × f_FOV` 逐层 closure |
| [x] | [01A.5 Signed reaction handoff](steps/01a5-signed-reaction-handoff.md) | reaction import/export 与 replay δE 诊断完成；import=0，残差留给后续 |
| [x] | [01B Compact isotope replay ledger](steps/01b-compact-isotope-replay-ledger.md) | isotope×target×generation status、parent outcome、transition |
| [x] | [01B.1 Replay semantics cleanup](steps/01b1-replay-semantics-cleanup.md) | lookup miss/cutoff 拆分；rate/replay/dE energy handoff |
| [x] | [02 Deterministic package auditor](steps/02-deterministic-package-yield-auditor.md) | GPU actual vs package exact vs TOPAS source；10-MeV occupancy audit |
| [ ] | [03 Lookup miss semantics/support](steps/03-lookup-miss-semantics-and-support.md) | 阶段 A/03B-0/03B-1 root-cause 已完成；03B-1R Be-6 数据补充、03B-2 support-aware rate 待完成 |
| [ ] | [04 MCS-only species/FOV](steps/04-mcs-only-species-fov.md) | step convergence、species-aware full-2GR、FOV acceptance |
| [ ] | [05 Stopping/range regression](steps/05-stopping-range-regression.md) | Be/Li explicit-table range 与 unrestricted deposition |
| [ ] | [06 Non-C12 straggling](steps/06-nonc12-straggling.md) | mean-preserving species-aware fluctuation |
| [ ] | [07 Full-cascade species closure](steps/07-full-cascade-species-closure.md) | Li/Be G0/G1/G2 birth→survival→transport→FOV |
| [ ] | [08 Package 修改门禁](steps/08-package-change-gate.md) | 仅在 job 485 + auditor 同向时修 construction cause |
| [ ] | [09 100/200/300 最终验收](steps/09-multienergy-final-validation.md) | 同一 physics 参数的多 seed 报告 |

### Step 03B-1 运行证据

2026-09-01 完成 `summary.csv` 作用域检查和 3.8 GB raw 全量流式审计。输出：
`plan/artifacts/cinel02-rate-package-census-e200-g1/projectile-campaign-audit.json`。

当前 package 的 102 个 source campaign 只有 17 个 projectile identity，不含 `Z4A6`；raw 中
`Z4A6` projectile interactions 为 `0`，但作为 C-12 direct child 的 `Z4A6` products 为
`27,186`，全部 role 0。结合 03B-0 的 `6Be+H/O` rate 全零、package nodes 为 0，
确定缺口来自 source campaign 未运行 Be-6 projectile exposure，而不是 compiler 丢弃已有
Be-6 projectile event。详见 [03B-1 step record](steps/03b1-be6-rate-coverage-root-cause.md)。

下一项是 03B-1R：先执行本地 200 MeV/u `GenericIon(4,6)` 50k smoke，再决定 5M
production；当前冻结 package 不覆盖。

## Step 03B-0 运行证据

2026-09-01 在沙盒外完成 rate/package census，输入为当前 200 MeV/u G1 基线使用的
`cascade_e400_rates.csv` 与 `research_hybrid_e200light107.cinpkg`，replay tolerance 为
`±0.51 MeV/u`。输出：
`plan/artifacts/cinel02-rate-package-census-e200-g1/census.json`、`census.csv` 和 `RESULTS.md`。

36 个显式 isotope×target rate group 全部存在，但只有 34 个有正 rate；package 也只有
34/36 个请求 group 有 global event nodes。`6Be+H1` 与 `6Be+O16` 的 rate sample 均为
445 个且全为零，package node 数均为 0；`7Be/9Be/10Be` 与 `6Li/7Li` 均有正 rate 和
package nodes。由此确定 Be-6 的 GPU secondary replay candidate=0 是输入 coverage 缺口，
不是 100k 抽样偶然。全局 3249 个正 rate sample 落在 package support 外，需结合实际
occupancy 在 03B-2 处理；本步骤不改变 runtime physics。详见
[03B-0 step record](steps/03b0-rate-package-coverage-census.md)。

## Step 01B 运行证据

2026-09-01 在本机 RTX 2080Ti、200 MeV/u、G1、100k、seed `2026095100` 完成。
Replay status（10 MeV/u bins）为 candidate/valid/lookup-miss/invalid/cutoff = `71524/71318/192/0/14`，
secondary generation-1 no-event `206/33512 = 0.615%`；按 isotope 最高为 2H `1.779%`、
3H `0.798%`、7Be `0.490%`、6Li `0.358%`。结果详见
[01B step record](steps/01b-compact-isotope-replay-ledger.md)。

### Step 01B.1 运行证据

2026-09-01 在本机 RTX 2080Ti/sm_75、200 MeV/u、G1、100k、seed `2026095100`
完成沙盒外 GPU 重跑。构建成功，CTest `2/2` 通过。输出：
`out/beam_200MeVu_cinel02_e200light107_g1_transitiondiag4_100k_xy04/energy_ledger.json`。

五状态总计：`candidate=71524`、`valid=71318`、`lookup_miss=192`、
`invalid=0`、`post_em_below_cutoff=14`；满足
`71524 = 71318 + 192 + 0 + 14`，且逐 cell partition 无异常。
分 isotope 的非零计数为：1H `5296/5290/6/0/0`、2H `8770/8614/153/0/3`、
3H `3133/3108/24/0/1`、3He `1445/1444/1/0/0`、4He `12487/12475/6/0/6`、
6He `61/61/0/0/0`、6Li `279/278/0/0/1`、7Li `263/263/0/0/0`、
7Be `204/203/0/0/1`、9Be `31/31/0/0/0`、10Be `49/49/0/0/0`、
8B `3/3/0/0/0`、10B `241/240/1/0/0`、11B `484/483/0/0/1`、
10C `30/30/0/0/0`、11C `499/498/0/0/1`、12C `38249/38248/1/0/0`
（字段顺序 candidate/valid/lookup_miss/invalid/cutoff）。

`E_rate - E_replay - dE` 的全局相对残差为 `1.98e-7`；最大 cell 相对残差
`1.20e-6`，来源是 device float atomic 累加，未改变物理结果。JSON layout 已更新为
`[18,2,3,40,5]`（10 MeV/u diagnostic bins），并输出 `rate_query_energy_MeV`、`replay_query_energy_MeV`、
`continuous_loss_to_collision_MeV`；generation 语义改为 `reaction_generation`。

该步骤只修正诊断语义，不修改 sampler/yield、rate、stopping、MCS 或 cascade physics。
Step 02 deterministic package auditor 已完成；下一步进入 Step 03 replay-support consistency。

## 提交切分

1. `feat(diag): add stratified secondary energy closure ledger`
2. `feat(diag): close signed secondary reaction handoff`
3. `fix(cinel02): preserve secondary transport semantics on replay miss`
4. `fix(mcs): apply species-aware scaling to full 2gr distribution`
5. `test(mcs): add mono-ion isotope lateral and range regressions`
6. `feat(straggling): add non-c12 secondary loss fluctuations`
7. `test(cascade): add 200mevu species closure regression`

sampler/package commit 不在预定队列中；必须先通过 Step 08 门禁。

## 更新规则

- README 是唯一进度入口；只执行第一个未完成步骤。
- 只有步骤文件的所有退出条件通过并记录证据时才标记 `[x]`。
- 每次记录 commit、canonical config、seed、histories、TOPAS job ID、package/rate SHA256、FOV/scorer header、输出路径和失败项。
- artifacts 保留在本地/15 TB 数据盘，不将 raw/3D dose 提交到 Git。
- 400 MeV/u 不阻塞当前验收；不用它促使 100–300 MeV/u 添加经验 scale。
