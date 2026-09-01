# 07：G1/G2 Secondary Cascade Physics

## 目标

每个secondary projectile使用自身合格package，修复G2的He消耗、重残片过量和能量残差。

## 实施步骤

1. G1/G2共享代码，仅由maximum generation控制。
2. secondary hazard严格查询实际Z/A、H/O和E/A shard；禁止C12/近邻isotope fallback。
3. 输出每代reaction、birth、dose、coverage、destruction/regeneration matrix。
   - [x] GPU runtime 已输出 generation × parent-Z × child-Z 的 count、kinetic energy、parent outcome 与 incident energy。
4. 对He3/4/6统计G0 birth、G1/G2 destruction、regeneration、cutoff、unsupported。
5. 400 MeV/u Secondary C/B +10%：检查重反应yield、parent continuation、event reuse。
6. 300 MeV/u Be -13%：按isotope检查channel。
7. 比较G0→G1、G1→G2增量，不以total抵消判断正确性。

## 2026-09-01 证据

- 配置：`config/beam_200MeVu_cinel02_e200light107_g2_transitiondiag_100k_xy04.yaml`，seed `2026095100`，100k。
- G2 species：Be -8.930%、Secondary C -2.741%、B -0.186%、Li -2.618%、He +1.525%、Z1 +1.022%、Primary C -0.492%。
- G1→G2 使 Be 再恶化 0.53 个百分点，因此 generation 截断不是 Be 缺口主因。
- G1 Be：309 reactions，incident 203.530 GeV；outgoing Be 70.965 GeV，net -132.565 GeV；C→Be 22.278 GeV，B→Be 32.961 GeV。
- G2 Be：11 reactions，incident 7.111 GeV；outgoing Be 2.009 GeV，net -5.102 GeV；该增量方向与 IDD 恶化一致。
- 下一判据：分别验证 Be-7/9/10 的 H/O reaction rate；随后按 target、50 MeV/u 能区拆分 Be destruction 与 C/B regeneration channel。
- [x] Water rate-only 100k（TOPAS job 474）：显式离子表修复后，100/150 MeV/u reaction fraction差为 -1.79%到+0.08%；50 MeV/u为 -6.52%到-3.54%。旧C-12 primary表导致Be射程短约56%，已删除核输运中的Z²/36 scaling与C-12 fallback。
- [x] 200 MeV/u G1回归与修复前逐项一致：Primary C -0.492%、Secondary C -2.626%、B +0.187%、Be -8.396%、Li -2.353%、He +2.028%、Z1 +1.350%。因此当前Be残差不是显式停止表加载问题。
- [x] 新增紧凑的 generation×target×50 MeV/u Be channel ledger。G1 Be incident：H 58.069 GeV、O 145.461 GeV；Be→Be：H 6.277、O 5.248 GeV；B→Be：H 20.340、O 12.620 GeV；C→Be：H 16.336、O 5.942 GeV。O占Be反应入射能量71.5%，但仅占C/B再生能量33.5%，为首要package channel审计对象。
- [x] TOPAS full-cascade 100k（job 477，`IncludeSecondaries=TRUE`）已完成并解析：50 workers、73,669 interactions、552,119 products。G1 Be incident 为 H 51.255 GeV、O 140.824 GeV；全部 Be output 为 H 21.173 GeV、O 27.786 GeV。GPU/TOPAS 的 G1 Be 净动能变化分别为 -136.77/-143.12 GeV，仅差约4.4%，否决“统一 Be 过度销毁”作为 -8.4% dose 缺口主因。
- [x] target 分解显示抵消：GPU H 上 B/C→Be 为 20.340/16.336 GeV，TOPAS 为 8.912/9.494 GeV；GPU O 上为 12.620/5.942 GeV，TOPAS 为 15.625/7.721 GeV。下一步改查 G0/G1 的 Be-7/9/10 composition、birth spectrum、输运 dose attribution，不做统一 rate 修正。
- [x] Be-6 单变量显式停止表 A/B：CSV 已有 Geant4 Be-6 表，但旧 loader 硬编码17种并在 transport 起点终止 Be-6。加入第18种显式表后，Be 积分差由 -8.396% 改善为 +2.625%，NRMSE 3.714%→2.495%；其他六类逐位不变，total -0.515%→-0.463%。这证明 Be-6 缺失是主因，但稳定输运略过修复；下一步用 TOPAS Be isotope charged-origin 3D scorer确定 Be-6 衰变/子代归属。
- [x] TOPAS isotope-origin 3D reference（job 478）：Be-6/7/9/10/other 为 0.236187/1.489672/0.207658/0.220092/0.003857 Gy，isotope sum 对 aggregate closure 为 -2.0e-9%。GPU Be-6 新增 0.237795 Gy，差 +0.681%，证明显式 Be-6 输运正确；GPU 旧有 Be-7/9/10 合计相对 TOPAS non-Be6 为 +2.865%，是剩余总Be +2.625%的来源。
- [x] GPU isotope 3D scorer：Be-6/7/9/10 分别为 +0.681/+2.781/-0.318/+8.234%，峰位差 +4.0/-2.5/+1.0/+3.0 mm，isotope sum 对aggregate closure 2.4e-8%。不同isotope峰位偏移方向相反，剩余问题指向isotope-conditioned birth spectrum/depth而非统一rate或stopping修正。
- [x] Isotope birth ledger与cosine：C12+O16→Be10 count GPU/TOPAS=141/130，mean E/A=68.65/84.32（GPU软18.6%）；H通道142.05/139.01正常。G0 mean cos=0.689/0.825，过度前向假设被否决。下一步仅针对O16按parent collision-energy cell审计，不做全局Be修正。

- [x] Parent-energy cell审计：GPU/TOPAS/package的C12+O16反应parent mean E/A逐箱一致；package Be10谱接近TOPAS，但旧runtime nearest-single回放系统性软化。research107含3,663,101 events和3,573,494个精确能量节点，平均1.025 events/node。
- [x] Runtime改为±0.51 MeV/u窗口内对完整correlated events按事件数均匀抽样；host/device语义与测试同步。100k后Be10聚合mean E/A 68.65→81.60（TOPAS 84.32），证明lookup根因修复。artifact：`plan/artifacts/cinel02-e200light107-g1-windowlookup-be-e200/analysis.json`。
- [ ] Window-event A/B同时暴露package isotope yield偏差：Be总量+4.844%，其中Be6/7/9/10为+10.388/+2.730/+13.496/+6.881%。下一步审计isotope×target×parent-energy条件yield/KE，不回退nearest-single，不做经验归一化。
- [x] 扩展 TOPAS raw 条件账本为 generation×target×parent-channel×Be-A×parent-energy，并修复旧脚本将B/C channel错误除以Be incident count及遗漏G0的问题。
- [x] 新增GPU G0 C12+O16 Be-6/7/9/10逐parent-energy birth count/KE诊断；CTest 2/2通过，同seed 100k GPU耗时7.38s，IDD与window-event基线一致。
- [x] 定位Be9首要偏差：50--100 MeV/u package yield较TOPAS高33.8%，每反应Be9总KE高29.1%；GPU实际yield高45.0%（64 births）。
- [x] Source campaign拆分：50--100 MeV/u反应98.3%来自200 MeV/u source，跨campaign yield一致；否决mixed-source weighting为偏差根因。
- [ ] 统计资格：100k TOPAS该格仅43个Be9 births，与package差异约2σ。TOPAS 1M full-cascade job 485运行中；完成前不得修改sampler或isotope yield。
- [ ] Be6并非G0 O16 yield系统偏高：window sampling后G0总birth KE约比TOPAS高3%，但dose高10.4%。需分离不稳定Be6/后代归属、birth angle/depth和显式离子输运的dose-per-birth，不能恢复旧软谱制造抵消。
- [x] Dose-per-birth分解：Be6 birth KE +3.82%但dose +10.39%，dose/KE +6.33%；其余Be isotope也有+2.46%到+6.12%。Be6 G0更前向（cos 0.791 vs 0.735），下一步统一检查angle/MCS/FOV escape，不做isotope专用scale。

## 验收

- [ ] 所有secondary query使用qualified cell。
- [ ] coverage miss的dose contribution为零。
- [ ] G2不使He、Be、Z1相对G1恶化超过1个百分点。
- [ ] generation-wise residual ≤0.5%。
- [ ] G2 total若比G1差>0.2个百分点，G1保持生产默认。
