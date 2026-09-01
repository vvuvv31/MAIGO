# 02：Package Coverage Auditor 与资格认证

## 目标

区分“文件可读”和“物理覆盖合格”，量化每个 projectile/target/energy cell 的统计有效性。

## 实施步骤

1. 扫描 CINPKG03 interactions/rates，建立 `(Zp,Ap,Zt,At,Ebin)` 五维矩阵；Ebin 宽 1 MeV/u。
2. 每 cell 输出 event count、rate、multiplicity、species yield、birth E/A、parent survival、local/neutral/unsupported energy。
3. 输出 bootstrap 95% CI、effective sample size 和 event reuse 上限。
4. 从 G0/G1/G2 生成 demand matrix：hazard、lookup、hit/miss、候选数、dose-weighted coverage。
5. manifest 写入 qualification、coverage gaps、SHA256、TOPAS/G4/cuts/provenance。
6. loader 默认拒绝 `statistically_qualified=false`；研究模式须显式 opt-in 并打印缺口。

## 资格门槛

- C12、p、He4 每 H/O/energy cell ≥1000 events。
- 其余 14 isotope 每 cell ≥500 events。
- 主要 yield 相对 CI ≤2%，mean birth E/A 相对 CI ≤1%。
- G1 demand-weighted coverage ≥99.99%，G2 ≥99.9%。
- dose contribution >0.1% 的 projectile 不允许 coverage miss。

## 当前执行记录（2026-09-01）

- 修复 GPU hazard histogram generation：primary interaction=0，首次 secondary cascade=`frag.generation + 1`；物理输出不变。
- 200 MeV/u、100k、G1 demand：primary 38,012 hazards，generation-1 cascade 33,486 hazards。
- demand artifacts：`plan/artifacts/cinel02-hybrid-pilot/pilot_complete_coverage_e200_g1_demand_genfix.json`、`e200_g1_demand_priority.csv`。
- hybrid pilot：2361 cells，777 qualified，1584 unqualified，22 gaps。
- C-12 production 首次提交 job 427 在 TOPAS 初始化失败：exposure scorer `ProjectileZ/A=0/0`（exit 134），无物理 histories；已修复模板替换并改用新 UUID/root。
- 有效提交：Slurm job 433，array 0-5%3，10/25/50/100/200/400 MeV/u，每点 5M histories；60 CPU/40G each，峰值 180 CPU/120G；输出 `/mnt/sda/wuwei/cinel02-c12-production-5m-v2`。
- 2026-09-01 完成审计：10/25/50/100/200/400 MeV/u 均记录 `Total number of histories: 5000000`，且各有 60 个非空 worker `.cinel02`；`.err` 仅含 `time` 统计。raw 实际位于各能点的 `raw/<uuid>/`，不能用能点目录顶层 glob 判断缺失。
- 已将 102 个 pilot campaign 与 6 个 C12 production campaign 流式合并为 `/mnt/sda/wuwei/cinel02-c12-production-5m-v2/combined108/research_hybrid_c12prod108.cinel02`：8,103,953 interactions、74,855,333 products、9,409,144,728 bytes；合并峰值 RSS 12 MB。
- 新 hybrid coverage：2362 cells，906 qualified，1456 unqualified，22 gaps；相对 pilot 的 777 qualified 有改善。C12 从 15/144 qualified、67,696 events 提升为 144/145 qualified、6,825,209 events；唯一失败为 C12+O16 的 1--2 MeV/u cell（1 event）。
- 200 MeV/u G1 demand：G0 conservative coverage 30.75%，G1 2.93%；C12 production 已解决 primary package 密度，但 p/d/t/He 等 cascade projectile 仍只有 pilot 统计。
- artifacts：`c12prod108_coverage_1mev.json`、`c12prod108_cells_1mev.csv`、`c12prod108_coverage_hybrid.json`、`c12prod108_coverage_hybrid_runtime17.json`、`c12prod108_coverage_e200_g1_demand.json`。
- 已构建受控中间包：pilot 102 campaigns + 200 MeV/u C12 5M campaign，共 3,219,536 interactions、27,888,674 products；沿用旧 research package 的 5000 MeV/u compatibility cell、minimum-events=1、unsupported gate=0.002。legacy 编译耗时 12m41s，MaxRSS 55.6 GB，证明完整 9.4 GB raw 必须使用 streaming compiler。
- 本机 RTX 2080Ti/sm_75 重新编译当前源码，CTest 2/2 passed；相同 seed 2026095100 的 200 MeV/u、100k、G1 GPU A/B 已完成。相对旧 pilot package：Primary C -0.492% 不变，Total -0.499%→-0.512%，B +2.246%→+0.183%，Li -2.967%→-2.368%，Secondary C +0.291%→-2.636%，Be -7.230%→-8.406%，He +1.009%→+1.981%，Z1 +1.490%→+1.506%。
- A/B artifacts：`plan/artifacts/cinel02-e200prod103-e200/analysis.json` 与 `species_idd_comparison.png`。结论：更多 C12 statistics 未破坏 rate/primary/total，但改变 first-interaction fragment composition；旧 pilot 的 Secondary-C/B/Be 接近程度部分是统计偶然，下一诊断层为 G0 birth yield/composition，而不是 dose normalization。
- G0 isolation（new C12 5M package, same seed）：相对完整 TOPAS，Secondary C +9.997%、B +19.127%、Be +5.366%、Li +18.386%、He +15.809%、Z1 -13.108%；开启 G1 后分别为 -2.636%、+0.183%、-8.406%、-2.368%、+1.981%、+1.506%。结论：G1 cascade 是必要物理；Be 的主要问题是 G1 destruction 过强或 regeneration 缺失，不应修改 C12 G0 yield。
- C12 raw birth audit：主要差异位于 O16、150--200 MeV/u。新 5M 相对旧 50k 的 Z6/Z5 yield 每反应约低 6.2%/7.0%；Z4 yield略高约3.1%，但 birth E/A低约6.9%且更不前向。旧 pilot 的接近包含统计偶然；未发现提取/ledger异常。
- job 439 无效：四项 TOPAS 均完成 5M histories，但使用 `GenericIon(1,1)` 等定义，只生成 exposure CSV，CINEL02 raw/contract 均为空；命名 light-ion process 未被 capture physics 包装，不能使用。
- job 443 有效完成：p/d/t/He4 各5M histories、各60个非空 raw和campaign contract；共10,900,181 interactions、68,021,753 products。hybrid coverage 409 cells，332 qualified、77 unqualified；50--200 MeV/u 高需求区基本 qualified，低能稀有反应仍不足。
- 为控制 legacy compile 内存，按 `(projectile,target,hybrid-bin)` 使用固定 BLAKE2b identity hash 做与产物无关的确定性缩减：保留443,565 interactions、2,162,562 products（约376 MB）；完整raw原样保留。
- 合并 C12-production+pilot+reduced-light-ion 为107-campaign package：3,663,101 interactions、30,051,236 products；legacy compile耗时14m48s、MaxRSS61.2 GB。
- 相同 seed 200 MeV/u、100k、G1 A/B：Z1 +1.506%→+1.350%，Li -2.368%→-2.353%，He +1.981%→+2.028%；Secondary C -2.636%→-2.626%，B +0.183%→+0.187%，Be -8.406%→-8.396%，Total -0.512%→-0.515%。light-ion package statistics不是Be/Secondary-C残差根因。
- artifacts：`plan/artifacts/cinel02-e200light107-e200/analysis.json`、`species_idd_comparison.png`、`plan/artifacts/cinel02-hybrid-pilot/lightion_prod5m_coverage_*.json`。
- 下一步：实现逐 generation、逐 parent-Z→child-Z 的 reaction/birth/destruction/regeneration ledger，重点检查 Be 由 C/B 生成及 Be→Li/He 的再反应；不再增加 light-ion statistics。streaming compiler仍为完整 campaign 构建前置项。

## 验收

- [ ] 当前 `cascade_e400` 被判为未认证并列出缺失 cells。
- [ ] 单元测试覆盖空/稀疏 cell、缺 H/O 和能量断层。
- [ ] runtime diagnostics 与离线 demand matrix 一致。
