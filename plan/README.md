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

- Development profile：`c2e25b832fbc015c611eb4d233da4a164abefd23`，显式 `cinel02_topas_compatibility_mode: true`；完整 manifest 为 [`baseline-e200-c2e25b8-topascompat.json`](baseline-e200-c2e25b8-topascompat.json)。
- Energy：200 MeV/u，100k histories，seed `2026095100`，G1；本机 RTX 2080Ti/sm_75。
- Package：`research_hybrid_e200light107.cinpkg`，window-event sampling `±0.51 MeV/u`。
- Grid/FOV：200×200×800，0.4×0.4×0.5 mm，80×80 mm；IDD 仅由 3D scorer 横向求和。
- Compatibility baseline species integral（独立 TOPAS/GPU seed，描述性非 CI）：Primary C +7.278%，Secondary C +3.327%，B +6.539%，Be -2.365%，Li +0.976%，He +5.183%，Z1 +4.207%，charged Total -0.569%。
- Be6 在 compatibility profile 中是 `non_transportable_prompt_decay`：produced=281、queued=0、discarded KE=130184.242188 MeV；relative Be6 dose gate 暂停，等待 matched depositing-track scorer。
- Secondary replay miss：`192/71524 = 0.268441%`；Li6/Li7、Be7/Be9/Be10 均为 0。
- TOPAS 1M full-cascade job 485 仍用于 Be9 统计门禁；sampler/yield/package 继续冻结。

## 优先级

当前路线：c2e25b8 compatibility rebaseline、03A attribution sanity check 与 03B-2B bounded source/compiler consistency 已完成；04A generation eligibility + H/O rate-coverage exposure audit 与 04B runtime optical depth × empirical survival 已完成；下一步进入 04C stopping residence/continuous optical depth。04B 未修改 rate、target selection 或 generation gate。03B-2B 的 192 个 runtime sparse-support miss 作为 correctness residual 携带，不调 rate/target mix；04A 暴露的 generation-blocked path 也先诊断，不直接改 G1 gate。

`P0 ledger correctness → P0 signed handoff → P0 compact isotope ledger → P0 replay semantics → P1 deterministic auditor → P0 Be6 compatibility policy/A-B → P0 compatibility rebaseline → P1 03A baseline attribution sanity check → P1 03B-2B bounded source/compiler consistency → P1 04A generation eligibility + H/O coverage exposure → P1 04B runtime optical depth × empirical survival → P1 04C stopping residence/continuous optical depth → P1 causal Be/Li fix → P2 light-ion accounting cleanup → P2 non-C12 straggling → P3 MCS shape → P4 100/300 regression`。

## 进度控制

当前完成度：**11/17（约 65%）**；Step 03 阶段 A、03B-0、03B-1 campaign provenance gate、03B-1R、compatibility rebaseline、03B-2A、03A attribution sanity check、Step 01B.1 和 03B-2B bounded source/compiler audit 已完成。由于新发现 Be-6 的基态寿命为 prompt scale，原先“补充稳定 Be-6 projectile campaign”的 03B-1R 已收缩为 TOPAS reference compatibility policy gate；reference 明确为无 daughter/无 deposit 的 StopAndKill，no-decay 数据只作诊断，不得编译进生产 package。

Step 01A ledger correctness、Step 01A.5 signed handoff、Step 01B compact isotope 诊断仪器和 Step 02 deterministic package auditor 已完成；04A 已完成 eligibility/coverage exposure 诊断，04C 仍未开始；04B-1 candidate-vs-tau、04B-2 blocked counterfactual hazard、04B-3 package-vs-runtime parent outcome 已完成；Step 03 阶段 A 已修复 replay miss 的 null-collision MCS semantics，03B-0/03B-1 已确定 Be6 coverage 缺口及 TOPAS prompt-unstable compatibility 语义，compatibility rebaseline 已切换为开发基线。03B-2A 与 03B-2B 已证明 transportable Be/Li 的 miss 为零，且当前 source/compiler/package/global-index 一致；192 个 runtime sparse-support miss 仅作为 correctness residual 携带，不直接改 runtime rate。p/d/He4 及当前 aggregate species residual 仍未修复，不能把 compatibility rebaseline 误记为物理收敛。

| 状态 | 步骤 | 主要产出 |
|---|---|---|
| [x] | [00 冻结 200 MeV/u 基线](steps/00-freeze-e200-baseline.md) | `baseline-e200-c2e25b8-topascompat.json`；旧 `becf880` 仅作历史对照 |
| [ ] | [01 Species 分层 ledger](steps/01-hierarchical-species-ledger.md) | `K_birth × f_dep × f_FOV` 逐层 closure |
| [x] | [01A.5 Signed reaction handoff](steps/01a5-signed-reaction-handoff.md) | reaction import/export 与 replay δE 诊断完成；import=0，残差留给后续 |
| [x] | [01B Compact isotope replay ledger](steps/01b-compact-isotope-replay-ledger.md) | isotope×target×generation status、parent outcome、transition |
| [x] | [01B.1 Replay semantics cleanup](steps/01b1-replay-semantics-cleanup.md) | lookup miss/cutoff 拆分；rate/replay/dE energy handoff |
| [x] | [02 Deterministic package auditor](steps/02-deterministic-package-yield-auditor.md) | GPU actual vs package exact vs TOPAS source；10-MeV occupancy audit |
| [x] | [03B-1R Be6 TOPAS compatibility policy/A-B](steps/03b1r-be6-prompt-decay-semantics.md) | TopasCompatKill、显式 discarded-kinetic sink、200 MeV/u 100k fixed-seed A/B 已完成；全局 inherited CINEL02 residual 仍待后续处理 |
| [x] | [03B-2A Occupancy-aware replay-support audit](steps/03b2-replay-support-occupancy-audit.md) | transportable isotope miss 分类完成；Be/Li miss=0；不改 runtime rate |
| [x] | [03A Baseline attribution sanity check](steps/03a-baseline-attribution-sanity-check.md) | 当前 HEAD 同 seed A/B 通过；稳定 species 仅约 1e-8% 变化，Be/Total 单独允许变化 |
| [x] | [03B-2B Bounded source/compiler consistency](steps/03b2b-source-compiler-consistency.md) | 174/18 provisional miss 分类已完成；source/compiler/index 无确认缺陷，不调 rate/target mix |
| [x] | [04A Generation eligibility + H/O rate-coverage exposure](steps/04a-generation-eligibility-rate-coverage.md) | secondary path、generation gate、H/O coverage、rate·ds 与 reaction outcome ledger |
| [x] | [04B Runtime optical depth + survival self-audit](steps/04b-runtime-optical-depth-survival.md) | candidate-vs-τ、generation-blocked counterfactual hazard、package/runtime parent outcome |
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

已完成 Be6 TOPAS compatibility policy/A-B；03A attribution sanity check 与 03B-2B bounded source/compiler audit 也已完成（仅 transportable isotopes）。Job 495 增强 scorer 已确认 TOPAS 对 GenericIon(4,6) 立即调用 RadioactiveDecay 并终止 parent；50k histories 没有任何可观测 daughter，RDM 数据目录也没有 z4.a6 衰变方案。该结果禁止稳定 Be-6 rate/package 补充，也禁止在 GPU 中猜测 alpha+p+p conversion。

## Step 03A 运行证据

2026-09-01 在当前 HEAD `2791bae` 重建 SYCL CUDA binary 后，以同一 200 MeV/u、G1、
100k、seed `2026095100`、package/rate、geometry 和 3D scorer 完成 compatibility=false/true
A/B。false 配置显式写入 `cinel02_topas_compatibility_mode: false`，true 配置为 `true`；
两份报告的 TOPAS 八个 3D scorer SHA256、histories、grid 和 scorer contract 完全一致。

- false analysis：`plan/artifacts/cinel02-baseline-attribution-e200-g1/off/analysis.json`
- true analysis：`plan/artifacts/cinel02-baseline-attribution-e200-g1/on/analysis.json`
- checker：`plan/artifacts/cinel02-baseline-attribution-e200-g1/sanity.json`，status=`pass`
- 稳定类别 GPU integral 最大绝对变化为 `2.74e-8%`（Primary C）；Secondary C、B、Li、
  He、Z1 均在 `0.01` percentage-point 门槛内。Be aggregate `-11.5263%`、charged total
  `-0.0577%` 是允许的 compatibility sink 变化。
- 旧 `baseline-e200-becf880.json` 没有绑定 TOPAS reference config/hash，且使用不同的
  `explicitsp` GPU config；因此旧 `becf880` 百分比仅作历史对照，新 compatibility baseline 才
  用绑定的 `total_species` TOPAS reference 作为开发基线。

该步骤只排除了 compatibility 开关导致的 scorer/analysis side effect，不构成物理收敛结论；
04B 已完成 runtime optical-depth self-audit；04A 显示的 G1 generation-blocked path 已用反事实
hazard 单独量化，下一步进入 04C stopping residence/continuous optical depth。

## Step 04A 运行证据

2026-09-02 在本机 RTX 2080 Ti/sm_75、200 MeV/u、G1、100k、seed `2026095100`
完成 secondary exposure smoke。实现只增加 diagnostics，不改变 rate、target selection、step size、
package、stopping、MCS 或 sampler。

- Ledger：`out/beam_200MeVu_cinel02_e200light107_g1_topascompat_100k_xy04/energy_ledger.json`。
- 汇总：`plan/artifacts/cinel02-exposure-e200-g1-topascompat/summary.json`。
- 16,994,005.1 mm total secondary path 中，13,921,212.1 mm 为 generation-eligible，
  3,072,793.0 mm（18.08%）被 G1 generation gate 阻断。
- eligible path 中 13,916,848.8 mm rate-pair covered，4,316.7 mm uncovered（约 0.031%）；
  6Li/7Li、7Be/9Be/10Be 的 coverage 分别为 99.942%、99.962%、99.954%、99.796%、99.939%。
- secondary candidate/valid/killed/continued = `33512/33306/33306/0`；与既有 replay status
  partition 一致。

结论：H/O rate coverage 对 Be/Li 基本完整，不能解释大剂量偏差；G1 generation-blocked path
则不可忽略，04B 已把 eligibility、counterfactual hazard 与 empirical candidate 分开校验，
不直接调大 `cinel02_max_secondary_inelastic_generations`。


## Step 04B 运行证据

2026-09-02：完成 04B-1 candidate-vs-tau、04B-2 blocked counterfactual hazard、04B-3 package-vs-runtime parent outcome。主要 Li/Be 的 blocked-hazard fraction 为 6Li 4.49%、7Li 2.05%、7Be 1.76%、9Be 4.90%、10Be 11.06%；candidate-vs-tau z-score 均在约 ±2σ 内。当前 occupied package cells 的 parent outcome expectation 均为 100% kill，GPU 为 33306 kill / 0 continue。04B 未修改 rate、target selection 或 generation gate，下一步进入 04C stopping residence 与 continuous optical depth。

## Step 03B-2B 运行证据

2026-09-01 在 compatibility baseline（200 MeV/u、G1、100k、seed `2026095100`）上完成
bounded source/compiler/package consistency audit。审计脚本为
`startup/package_tools/audit_cinel02_source_package_consistency.py`，输出
`plan/artifacts/cinel02-source-package-replay-consistency-e200-g1/report.json`；输入为
`research_hybrid_e200light107.cinel02`、`research_hybrid_e200light107.cinpkg` 和同一 campaign
`summary.csv`。

- raw 全量扫描 `3,663,101` records；package 为 `3,663,101` interactions、`30,051,236`
  products、`3,573,494` global energy nodes。34 个 source-backed isotope×target groups 全部
  `source_present_compiled`；仅 6Be 的 H/O 两组为预期 `source_missing`，因为 compatibility
  policy 将其标记为 non-transportable。`compiler_dropped_support=0`、
  `package_has_unbacked_support=0`、raw/package 不匹配均为 0。
- 03B-2A 的 192 个 occupied miss 中，174 个（20 cells）仍是 provisional
  raw/compiler-support-gap bucket，18 个（9 cells）是 provisional index/lookup-anomaly
  bucket；这些 cells 全部属于 `source_present_compiled` key，不能解释为 compiler 丢包。
- 对 package 全部 `3,573,494` 个 node 执行 `14,293,976` 个 exact node/window replay probes，
  `exact_replay_failures=0`。因此没有确认的 global-index、window-selection 或 `(Z,A)` lookup
  bug；不修改 package、rate、target CDF、tolerance 或 runtime support mask。
- 生产 package C++ inspection 与 Python census 一致：34 个 source-backed groups、
  `parent_survival_fraction=0`；6Be 不进入 transportable projectile/rate/replay coverage。

结论：03B-2B 是 bounded correctness cleanup，未找到可提交的 source/compiler/index 修复。
transportable Be/Li 的 replay miss 为 0；其余 192 个 miss 作为 runtime sparse-support
correctness residual 携带，不以 dose 改善作为验收。下一步进入 Be7/9/10、Li6/7 的
`track-length × hazard optical depth → empirical reaction survival → stopping residence`。

## Step 03B-1R 运行证据

2026-09-01 本机 TOPAS Job 495（200 MeV/u、50k、seed `2026099609`）完成增强 scorer 诊断。
50,000/50,000 条记录均为 `(Z,A)=(4,6)` 的 `decay_parent`，post-step process 为
`RadioactiveDecay`，step length 约 `10^-9 mm`，pre/post KE 均为 1200 MeV；`decay_daughter`
为 0，step deposit 为数值零。`RadioactiveDecay6.1.2` 中缺少 `z4.a6`，只有
`PhotonEvaporation6.1/z4.a6`。因此当前 TOPAS reference 的可观测行为是立即终止且无 daughter，
但三体衰变尚无证据。详见 [03B-1R step record](steps/03b1r-be6-prompt-decay-semantics.md)。

03B-1R 已确认 reference compatibility 语义，完成 TopasCompatKill policy、显式 sink 和固定 seed A/B；Be6 produced=281、queued=0、discarded=281，sink KE 与 generated KE 相对残差约 1.1e-7。全局 accounting residual 约 1.536% 为既有 CINEL02 residual，不是 compatibility mode 新增。
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
Step 02 deterministic package auditor、03B-2A、03B-2B bounded source/compiler audit、04A exposure audit 与 04B runtime optical-depth self-audit 已完成；下一步进入 04C stopping residence。

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

- README 是唯一进度入口；按当前路线执行第一个未完成且不被前置诊断阻塞的步骤，已分类但未收口的 light-ion residual 可携带到后续支线。
- 只有步骤文件的所有退出条件通过并记录证据时才标记 `[x]`。
- 每次记录 commit、canonical config、seed、histories、TOPAS job ID、package/rate SHA256、FOV/scorer header、输出路径和失败项。
- artifacts 保留在本地/15 TB 数据盘，不将 raw/3D dose 提交到 Git。
- 400 MeV/u 不阻塞当前验收；不用它促使 100–300 MeV/u 添加经验 scale。
