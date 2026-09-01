# CINEL02 Total/Species IDD 修复总计划

## 使用规则

- 本文件是唯一进度入口；实施前查看当前步骤，再进入对应 `steps/*.md`。
- 只有步骤文件全部验收并有证据时才能标记 `[x]`。
- 单 seed 100k 仅用于诊断；最终结论使用至少 5 个独立 100k seed。
- TOPAS 仅在 `wuwei@127.0.0.1` 经 `sbatch` 运行；GPU 仅在本机 RTX 2080Ti/sm_75 运行。
- 仅使用 3D scorer 横向求和；不以 FP32/FP64 切换或经验 dose scale 修复物理误差。

## 总体验收

- Total：积分和峰值差 ≤2%，峰位差 ≤0.5 mm，IDD NRMSE ≤1%。
- 七类 species：积分差 ≤2%，峰位差 ≤1 mm，IDD NRMSE ≤2%。
- Lateral core sigma 中位差 ≤2%，halo sigma 中位差 ≤5%。
- 分类 closure 可解释；逐 generation energy residual ≤0.5%。
- 100/200/300 MeV/u 为当前硬验收域；400 MeV/u 暂列为非阻塞外推诊断。绘图深度为 TOPAS total Bragg peak 的 1.2 倍。

## 进度控制

当前完成度：**2/11（18%）**。当前下一步：**等待并解析 TOPAS 1M job 485，判定 Be9 条件yield是否为真实偏差；并行设计 Be6 dose-per-birth/不稳定核后代归属 A/B，不修改 sampler/yield**。

| 状态 | 步骤 | 主要产出 |
|---|---|---|
| [x] | [00 基线与根因证据](steps/00-baseline-and-root-cause.md) | 固化 rate、G0/G1/G2、四能量 IDD、double-sigma 与 package 风险 |
| [x] | [01 Charged-origin scorer](steps/01-charged-origin-scorer.md) | TOPAS/GPU 使用相同 electron attribution |
| [ ] | [04 Event sampling 与能量重整](steps/04-event-sampling-and-reconciliation.md) | 移除 nearest-single-event replay，逐事件守恒 |
| [ ] | [05 Parent/local-deposit 账本](steps/05-parent-and-energy-ledger.md) | 消除 continuation、EM loss、local deposit 重复/丢失 |
| [ ] | [02 Package coverage auditor](steps/02-package-coverage-auditor.md) | 五维覆盖矩阵与 qualification |
| [ ] | [03 Package exposure 与重建](steps/03-package-exposure-and-build.md) | 17 projectile、H/O、0--400 MeV/u package family |
| [ ] | [06 G0 birth physics](steps/06-g0-birth-physics.md) | yield、birth energy、角分布和 isotope composition 对齐 |
| [ ] | [07 G1/G2 cascade physics](steps/07-g1-g2-cascade-physics.md) | 正确 secondary package 与 He destruction/regeneration |
| [ ] | [08 Neutral 与缺失核过程](steps/08-neutral-and-missing-physics.md) | neutral-origin、elastic/decay A/B 与 total closure |
| [ ] | [09 Stopping/straggling/MCS](steps/09-transport-and-lateral.md) | 17 isotope mono-ion 与 core/halo 验证 |
| [ ] | [10 多能量最终验收](steps/10-final-multienergy-validation.md) | 100/200/300 MeV/u G1/G2、5 seeds、完整最终报告；400 MeV/u 非阻塞外推 |

## 当前已知基线

- 当前优先范围为 100--300 MeV/u；400 MeV/u 不阻塞 runtime 与 package 修复验收，仅保留外推诊断。

- Rate-only：reaction fraction 0.180%，reaction CDF 0.461%，primary survival 0.304%，H/O 0.101%。
- 200 MeV/u Z1 已统一为 p+d+t；multi-projectile research package、current-Z 核分类下 G1 Z1 100k 积分差为 +1.49%。
- G1 total 积分差：100/200/300/400 MeV/u 为 -0.49/-0.42/-1.40/-4.14%。
- G2 为 -0.49/-0.49/-1.58/-4.31%，未改善 total，保留与 G1 并行诊断。
- 400 MeV/u G2：Primary C +11.92%、Secondary C +10.54%、B +10.04%、He -16.46%。
- core sigma 基本匹配；400 MeV/u halo sigma 中位约宽 3.79%。
- 当前 `cascade_e400` 与 `research_hybrid_102` 均为 `statistically_qualified=false`。hybrid pilot 2361 cells 中仅 777 qualified、1584 unqualified、22 gaps；不能用运行时 isotope fallback 掩盖 yield 统计不足。
- rejected A/B：全窗口 event sampling 使 B/Li/Z1 更差；Z≤6 域外 isotope 全量 EM transport 使 Be/Li 改善但 C/B 超量，均已撤销。
- 200 MeV/u 100k `research_hybrid_e200light107`、current-Z 核分类：G1 Primary C -0.49%、Secondary C -2.63%、B +0.19%、Be -8.40%、Li -2.35%、He +2.03%、Z1 +1.35%、total -0.52%；G2 Be -8.93%、Secondary C -2.74%，说明增加 cascade 深度不能修复残差。
- TOPAS full-cascade job 477：50 workers、73,669 interactions、552,119 products。G1 GPU/TOPAS Be 净动能变化为 -136.77/-143.12 GeV；统一 Be destruction-rate 偏差不足以解释 -8.40% dose，下一诊断转向 isotope composition/spectrum/scoring。
- Be-6 显式停止表单变量 A/B：Be -8.396%→+2.625%，NRMSE 3.714%→2.495%；其余 species 不变。Be-6 缺失已确认为主要根因，但需用 TOPAS isotope-origin 3D scorer处理短寿命核素归属，不能经验缩放。
- TOPAS job 478 isotope-origin 3D closure：Be-6/7/9/10/other 分别为 0.236187/1.489672/0.207658/0.220092/0.003857 Gy，和 aggregate 2.157466 Gy 在 2.1e-9% 内闭合。GPU Be-6 增量 0.237795 Gy，仅比 TOPAS Be-6 高0.681%，故显式稳定 Be-6 输运正确；剩余总Be +2.625%来自旧有 Be-7/9/10 合计 +2.865%，下一步拆分 GPU isotope dose。
- GPU Be isotope 3D scorer closure 2.4e-8%。Be-6/7/9/10 积分差为 +0.681/+2.781/-0.318/+8.234%，峰位差为 +4.0/-2.5/+1.0/+3.0 mm；Be-9积分已匹配而各峰位方向不同，否决统一Be stopping/rate修正。下一步对比 generation×target×parent-channel×Be-A 的 birth count/energy/depth，重点检查Be-10 package replay correlation。
- Birth ledger：G0 C12+O16→Be10 GPU/TOPAS count=141/130，mean E/A=68.65/84.32 MeV/u（软18.6%）；C12+H→Be10为142.05/139.01，正常。G0 Be10 mean cos GPU/TOPAS=0.689/0.825，否决过度前向；G1虽为0.997/0.952但占比较小。根因收敛到O16 Be10的yield+soft-spectrum，下一步按parent collision E/A定位错误package cells。
- Parent-energy cell诊断：GPU/TOPAS/package的C12+O16 parent mean E/A在四个50 MeV/u箱内一致到0.41 MeV/u以内，但旧nearest-single lookup的Be10 mean E/A为7.59/29.01/58.77/93.52，package为11.48/35.42/72.21/111.61。根因是3,663,101 events形成3,573,494个精确浮点节点，随机数几乎失效。改为±0.51 MeV/u窗口内按完整event均匀抽样后，Be10聚合mean E/A由68.65改善至81.60（TOPAS 84.32）。
- Window-event A/B：Primary C -0.492%、Secondary C -1.452%、B +0.963%、Be +4.844%、Li -3.668%、He +0.767%、Z1 +1.751%、total -0.511%。Be isotope为Be6 +10.388%、Be7 +2.730%、Be9 +13.496%、Be10 +6.881%；旧单节点lookup曾偶然压低Be6/Be9，下一步修package条件yield，禁止退回确定性单节点以制造抵消。
- 2026-09-01 Be isotope 条件审计：修正 TOPAS raw analyzer 中 B/C channel 错用 Be incident denominator 且排除 G0 的缺陷；新账本覆盖 generation×target×parent×A×parent-energy。artifact：`plan/artifacts/be-isotope-conditional-audit-e200/`。
- 新增只读 GPU diagnostic slots 1604--1667，记录 G0 C12+O16 的 Be-6/7/9/10 × 8 parent-energy bins 的 birth count/KE；同 seed 100k IDD 完全复现 window-event 基线（total -0.511%，Be +4.844%）。
- C12+O16→Be9 在 50--100 MeV/u 的 package/TOPAS yield 为 0.01231/0.00920，每反应总 KE 偏高29.1%；GPU实际条件yield偏高45.0%（64 births，仍受单seed统计限制）。这是Be9 +13.5%的首要结构候选。
- Source-campaign audit否决能量campaign混合：该格242,173个package反应中98.3%来自2400 MeV C12 source；1200/2400/4800 MeV source的Be9 yield为0.01275/0.01230/0.01389，无系统分裂。100k TOPAS该格仅43个birth，package为2982个，当前差异约2σ，不足以授权channel重权。
- 已提交TOPAS 200 MeV/u full-cascade 1M统计任务job 485（50 CPU、50G、seed 2026091001、UUID `...920003`），用于把C12+O16 50--100 MeV/u Be9条件yield统计误差从约15%降到约5%；完成前冻结sampler/yield。
- Dose-per-birth审计：GPU/TOPAS Be6 birth KE差+3.82%，origin dose差+10.39%，故dose/birth-KE高6.33%；Be7/9/10亦高2.46/3.47/6.12%。新GPU G0 Be6 mean cos=0.791，TOPAS=0.735，有限80 mm FOV的角分布/MCS/逃逸差异为共同候选，不实施Be6专用dose修正。
- Be6 的 G0 birth KE 从旧 nearest-single 相对TOPAS -10.5%改善到window sampling约+3%，但dose仍+10.4%；旧match属于软birth谱与transport的抵消。下一步并行检查source-campaign mixing以及Be6不稳定核/后代归属的dose-per-birth，禁止用yield缩放补偿transport。

## 更新规则

每完成一步，在步骤文件末尾记录配置、seed、TOPAS job ID、输入 SHA256、输出路径、核心指标和失败项；随后更新本文件复选框、完成度和“当前下一步”。
