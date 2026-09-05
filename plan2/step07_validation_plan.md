# Step07 验证方案（冻结，跑数前锁定，事后不得放宽）

> **审查更正：本版实验设计无效，不得用于 section-0 缩放结论。**
> 下方保留原文用于追溯，不是修正后的执行配置。原文混淆密度公式区间
> [-1000,-98) 与材料 section 0 [-1000,-950)，并遗漏 DensityCorrection。
> HU=-550/-100 分别属于材料 section 1/2；既有数据只可作跨材料诊断。
> 新实验必须另建修正版设计，先执行 tools/audit_schneider_response_scope.py
> 校验 HU。可核查 -1000/-975/-951（全部 section 0）；密度分别约
> 0.0113161/0.0393235/0.0662102 g/cm3。不能直接复用本版“16 倍”结论。

> 范围：C12、Schneider section 0、150–225 MeV/u。训练节点 150/200/225；
> held-out 175（永不参与拟合；本步不做任何拟合，只做独立性检验）。
> 密度：section 0 内三 HU，取值来自 parser 真实边界（硬编码禁止）。

## 1. HU 取值依据（`data/HUtoMaterialSchneider.txt` + `src/ct_grid.cpp:293`）

- Section 0 密度区间：HU ∈ [-1000, -98)，
  rho = 0.00121 + 0.001029700665188 × (HU + 1000)。
- 低端：HU=-1000 → rho 0.00121（既有 G0 数据复用，不重跑）。
- 居中：HU=-550 → rho 0.46458（材料仓 [-950,-120)，成分不同）。
- 近上边界：HU=-100 → rho 0.92794（仍 < -98，属 section 0）。
- 材料名规则来自 TOPAS `TsImagingToMaterialSchneider.cc:213`
 （`PatientTissueFromHUNegative550` / `...Negative100`），
  随 `PreLoadAllMaterials=True` 预加载存在。

## 2. 运行矩阵（各 12 histories，G0 几何，v2 Binary，seed 917001）

| case | 能量 | HU | 说明 |
|---|---|---|---|
| e200_hu-1000 | 2400 MeV | -1000 | 已有 v2_new（job 2341），复用 |
| e150_hu-1000 | 1800 MeV | -1000 | 训练节点 |
| e175_hu-1000 | 2100 MeV | -1000 | **held-out，只验不拟** |
| e225_hu-1000 | 2700 MeV | -1000 | 训练节点 |
| e200_hu-550 | 2400 MeV | -550 | 密度中 |
| e200_hu-100 | 2400 MeV | -100 | 密度高 |

资源：每作业 2CPU/4G，共 5 新作业 10CPU/20G（≤192/160G）。
磁盘：每片 ~180 MB，预算内；达预算停交不删数。

## 3. 观测量与 ROI（冻结）

- O1 逃逸比 esc/(dep+esc)；O2 可迁移比 delta_dep/total_dep；
  O3 birth/parent-loss；O4 匹配父 KE 带内逃逸比（带：2300/2350/2400/2450，
  随能量点整体平移，以各 case 实测 matched KE 分位数为准记录，不硬套）；
  O5 联合直方图 L1（相对 e200_hu-1000）；O6 投影纵向/径向量化。
- ROI：全 slab 为主；50 mm 内缩内部子集为辅。
- 误差方法（冻结）：以 12 个独立 history 的逐事件值计均值±SD/√12
  （逐事件家族 esc/dep 取自 audit `per_event_residuals`）；roots/steps
  相关，不计为独立样本。

## 4. 判定阈值（冻结，事后不得放宽）

- T1 密度：固定 200 MeV/u 下三 HU 任一 O1/O2 相对差 >20% →
  “检出密度依赖，单密度表覆盖 section 0 被拒”；≤20% →
  “无反证，但 12 histories 下仍 INCONCLUSIVE，不支持制表”。
- T2 能量：held-out 175 的 O1/O2 落在 150–225 线性插值 ±3σ_event 内为
  “插值相容（弱证据）”，之外为“非线性/外推风险，禁复用一维插值”。
- T3 1/rho 缩放：不预设成立；分别检验 O1/O2/O5/O6 与 1/rho 预测的偏离，
  任一项系统性偏离（同号×三带）即判“缩放假设不成立”。
- 门禁（硬）：逐片能量闭合 1e-3、零 overflow、v2.1 bundle 验证；任一失败
  整批判 BLOCKED。统计不足一律 INCONCLUSIVE，永不 PASS。

## 5. 禁止事项

- held-out 175 不得参与任何参数拟合；不得看结果后调阈值；
  不得平均分片分位数；不得删远尾；不得调 scale/beam/MCS/nuclear 凑数。
