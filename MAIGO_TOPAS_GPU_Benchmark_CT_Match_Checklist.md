# MAIGO 发文前 TOPAS/GPU Benchmark 与 CT Match 清单

> 适用范围：`vvuvv31/MAIGO` 的 `reconstruct` 分支。目标是验证 MAIGO 的 GPU condensed-history 碳离子输运在声明的能量、材料、几何和计分范围内与 TOPAS/Geant4 一致，而不是宣称其等价于完整 Geant4 physics list。

## 0. 论文中必须先锁定的验证协议

| 项目 | 必须记录/固定的内容 | 建议写法或输出 |
|---|---|---|
| 代码版本 | MAIGO commit SHA、分支、编译器、oneAPI/SYCL、CUDA plugin、编译选项 | Supplementary Table S1 |
| 参考端版本 | TOPAS、Geant4、HadronLET extension 版本与 physics modules | TOPAS 4.2.p3 / Geant4 11.3.2（若实际如此） |
| 物理列表 | TOPAS 的 7 个 modules；MAIGO `best/medium/fast` 的精确开关 | 配置文件随稿公开 |
| 输入同源 | 粒子数、能谱、spot 坐标/权重、束斑、发散、几何、CT grid、材料映射完全一致 | 自动生成 input manifest |
| 剂量定义 | 两端均为 dose-to-medium；体素质量、归一化、单位转换一致 | 不允许为每个病例单独重标定 |
| LET 定义 | dose-averaged LET、纳入粒子种类、cutoff、restricted/unrestricted stopping power 一致 | 同时报告 primary-C12 与 all-hadron LETd |
| 随机统计 | 每个关键场景至少 3–5 个独立 seed；报告均值和 95% CI | 先做重复性，再做代码间差异 |
| 后处理 | 相同 grid；若必须重采样，固定插值方法并同时保留原始 grid 结果 | 禁止只展示平滑后曲线 |
| 盲法阈值 | 在查看最终 CT 结果前锁定 gamma、range、dose、LET 接受标准 | protocol/README 中预注册 |
| 校准参数 | `dose_output_scale`、`straggling_scale` 等必须用独立 calibration set 确定 | validation 病例不可再次调参 |
| 队列完整性 | charged/cascade/neutral queue overflow 必须为 0；同时报告能量闭合 | 任一 overflow 非零则该运行不可作验证 |

## 1. TOPAS/GPU 必做物理 Benchmark（按优先级）

### A. 最小可发表集：必须完成

| 编号 | 场景与扫描矩阵 | 对比量 | 建议统计/接受标准 | 推荐展示 | 状态 |
|---|---|---|---|---|---|
| A1 | **均匀水中单能 pencil beam**：至少 100、200、300、400 MeV/u；建议覆盖模型最低/最高有效能量 | IDD、入口平台、Bragg peak、fragment tail、R80/R50、峰宽；横向 σ/FWHM 随深度 | R80 差 ≤1 mm（或 ≤1% range）；高剂量区局部剂量差 ≤2%；1D gamma 1 mm/2% 与 2 mm/2% 均报告 | 每个能量一张 IDD ratio panel；range error vs energy 汇总图 | ☐ |
| A2 | **束斑与多重散射**：同一能量在入口、mid-range、peak 前后、tail 取横向 profile | lateral σ、FWHM、80–20% penumbra、halo | σ/FWHM 差 ≤1 mm 或 ≤5%；不能只比较中心轴 | 2D dose map + 4 个深度横向 profile + σ(z) | ☐ |
| A3 | **能损涨落/射程展宽**：monoenergetic 与真实 energy spread（至少 0、0.5%、1%） | peak width、distal 80–20%、R80 方差 | 各 seed CI 重叠；展宽趋势单调且与 TOPAS 一致 | distal fall-off 放大图 | ☐ |
| A4 | **核反应与碎片尾**：水中高统计单能束，至少 200 与 400 MeV/u | primary attenuation、核反应率、C/B/Be/Li/He/p 分物种剂量或 fluence、tail dose | 反应率和主碎片积分差建议 ≤5%；tail 单独评价，不能被全局 gamma 掩盖 | 分物种深度曲线、tail ratio、能量闭合表 | ☐ |
| A5 | **LETd 水箱**：与 A1 同能量；primary-C12 和 all-hadron 分开 | LETd(z)、peak 位置/幅值、entrance/peak/tail ROI；原始 numerator/denominator | 高剂量区 LETd median error 建议 ≤5%；峰值/尾部单列，不在低剂量噪声区用无限相对误差 | Dose+LET 联合图、LET difference/ratio、ROI 箱线图 | ☐ |
| A6 | **异质层状 phantom**：水→肺→水、软组织→骨→软组织、含空气腔；正入射与至少一个斜入射 | 界面剂量、range shift、lateral spread、LETd、核碎片尾 | distal range ≤1–2 mm；界面前后分别做 gamma/差值；报告材料 MCS 开/关 | 材料条带 + depth dose/LET；界面局部放大 | ☐ |
| A7 | **SOBP / 多能量层**：临床宽度的 SOBP，先单轴再 2D 多 spot | plateau uniformity、proximal/distal edge、range、LETd 梯度 | plateau 均匀性差 ≤2%；3D gamma 2%/2 mm（10% cutoff）≥95%，并附 3%/3 mm | SOBP depth dose、LETd、gamma map/histogram | ☐ |
| A8 | **spot plan 解析与批处理一致性**：TOPAS spot file、TPS source；batched vs sequential | 总剂量、逐 spot 权重、坐标变换、能量层、MHD/CSV 一致性 | batched 与 sequential 在 MC 统计容差内；同 seed 的实现一致性需给最大/均方差 | 小型已知答案 spot pattern + difference map | ☐ |
| A9 | **数值收敛性**：step = 1.0/0.5/0.2/0.1/0.05 mm，relative energy loss、cutoff、cascade generation、queue capacity | R80、峰值、tail、LETd、运行时间 | 选定 production 参数后，继续加严造成的关键量变化 <0.5–1%；overflow=0 | accuracy–runtime Pareto 图 | ☐ |
| A10 | **统计收敛与重复性**：10⁵、10⁶、10⁷（必要时更高）histories，≥5 seeds | voxel uncertainty、R80/IDD/LET CI、gamma CI | 差异必须相对联合 MC 不确定度解释；最终对比不低于 10⁷ 或达到预定 uncertainty | error vs histories 的 log-log 图 | ☐ |
| A11 | **能量守恒与审计量**：所有上述场景 | initial、deposited、escaped、beamline removed、untracked/residual、overflow energy | 闭合残差设硬阈值（建议 <0.1%，或给出合理物理解释）；所有队列 overflow=0 | 每类场景一行的 balance table | ☐ |
| A12 | **端到端性能**：warm-up 后同一 GPU，1/10/100 M histories；单 spot、SOBP、CT | wall time、kernel time、初始化/I/O、histories/s、voxels/s、显存峰值、能耗可选 | 报中位数与波动；同时给 TOPAS CPU 核数/线程与硬件，避免只报含糊“×倍加速” | throughput scaling、breakdown、accuracy–speed 表 | ☐ |

### B. 强烈建议：显著增强审稿说服力

| 编号 | 场景 | 目的与关键输出 | 状态 |
|---|---|---|---|
| B1 | 不同 beam size/divergence 与 off-axis spot | 验证相空间和几何变换；展示 spot centroid、σx/σy、旋转/平移误差 | ☐ |
| B2 | CT 材料标定 phantom（air/lung/soft tissue/bone） | 分离 HU→材料/密度映射误差与输运误差；报告 WET 与 range shift | ☐ |
| B3 | 物理消融：material-MCS、secondary straggling、neutral transport 分别开/关 | 量化每项对 dose、LET、tail、速度的影响；对应 MAIGO 已有运行时开关 | ☐ |
| B4 | `best/medium/fast` 三 profile | 给出 accuracy–runtime–memory trade-off，明确 fast 不用于最终 LET 结论 | ☐ |
| B5 | reaction/cascade package 泛化 | package 内插能量与边界能量分开报告；训练/生成能量与验证能量严格拆分 | ☐ |
| B6 | GPU 可移植性 | 至少 NVIDIA 两代 GPU；若主张 SYCL 可移植，再加 Intel Arc/Level Zero，比较数值一致性 | ☐ |
| B7 | 重复构建/可复现性 | clean build、固定 seed、容器或环境锁文件、自动生成表图 | ☐ |

## 2. CT TOPAS/GPU Match：病例与实验设计

### 2.1 病例选择

| 分组 | 最低建议 | 选择原则 | 不可混用的角色 |
|---|---:|---|---|
| Calibration | 1–2 例或独立 phantom | 只用于固定 `dose_output_scale`、`straggling_scale`、HU/material 参数 | 不进入最终总体通过率 |
| Internal validation | ≥5 例；更理想 10–20 例 | 头颈、胸部/肺、盆腔或骨附近靶区；覆盖不同体型、射程和异质性 | 不再逐病例调参 |
| Stress cases | ≥3 个 | 空气腔、肺、厚骨、斜入射、field edge、小靶区、长射程 | 单独列出，不用平均值掩盖失败 |
| External/hold-out（若可行） | ≥1 中心或不同 CT protocol | 不同扫描仪/重建核/层厚/校准曲线 | 用于声明泛化能力 |

### 2.2 每例必须跑的配对条件

- [ ] 完全相同的 CT voxel grid、patient orientation、isocenter、gantry/couch angle、spot positions、energies、weights 和 histories。
- [ ] TOPAS 与 GPU 使用相同 HU→density/material 映射；另保存 material-label difference map。
- [ ] 单野分别比较，再比较总计划；总计划不能替代 per-field 诊断。
- [ ] dose-only 与 dose+LET 使用同一物理配置；记录 scorer 是否改变性能或内存。
- [ ] 至少 3 个 seed；TOPAS 与 GPU 的统计不确定度分别估计。
- [ ] primary-C12 LETd 与 all-hadron LETd 分开；低剂量/低 denominator voxel 设预先定义的 mask。
- [ ] 输出 raw deposited energy、dose Gy、LET numerator/denominator、物种剂量、队列/能量审计量。
- [ ] 同时运行 GPU `best` 基线和关键物理扩展消融；不要只展示调到最匹配的配置。

## 3. CT 结果应该怎样展示

### 3.1 每个代表病例的主图（建议 3 个病例：均匀、骨界面、肺/空气腔）

| Panel | 内容 | 展示要求 |
|---|---|---|
| A | CT + target/OAR + beam direction | 标注解剖平面、窗宽窗位、体素尺寸 |
| B–C | TOPAS dose 与 GPU dose | 相同色标、相同切面、显示 10–100% isodose |
| D | dose difference | 同时给绝对差 `GPU−TOPAS` (Gy) 与局部/全局相对差定义 |
| E | 3D gamma map | 2%/2 mm、10% cutoff 为主；3%/3 mm 为补充；失败 voxel 高亮 |
| F–G | TOPAS LETd 与 GPU LETd | primary-C12 与 all-hadron 至少各一组；相同色标 |
| H | LETd difference | 仅在 dose/LET denominator mask 内；避免背景噪声制造巨大相对差 |
| I | 选定 beam-path line profiles | 穿过软组织、骨/肺界面、Bragg distal edge；画 95% CI |
| J | DVH | target 与主要 OAR；TOPAS/GPU 同图 |
| K | LVH 或 dose–LET joint histogram | target/OAR 内 LETd 分布，或 dose–LET 2D histogram |
| L | 失败区域解释 | 将 gamma/LET outlier 与材料边界、低剂量、fragment tail 对齐 |

### 3.2 全病例汇总表

| Case | Site | Fields / spots | Histories | TOPAS uncertainty | Dose γ 2%/2 mm | Dose γ 3%/3 mm | R80 Δ (mm) | Target D95 Δ (%) | OAR Dmean/Dmax Δ (%) | LETd median / P95 error (%) | Runtime / speedup | Overflow |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Case-01 |  |  |  |  |  |  |  |  |  |  |  | 0 |
| Case-02 |  |  |  |  |  |  |  |  |  |  |  | 0 |
| Median [IQR] | — | — | — | — |  |  |  |  |  |  |  | — |
| Worst case | — | — | — | — |  |  |  |  |  |  |  | — |

> 不要只报告 mean/median：必须同时给最差病例、失败 voxel 的位置及原因。

### 3.3 推荐的定量指标

| 类别 | 主指标 | 补充指标/注意事项 |
|---|---|---|
| 3D dose | global gamma 2%/2 mm，10% max-dose cutoff | 同时给 1%/1 mm（探索）和 3%/3 mm（可比性）；报告 normalization 与 interpolation |
| Range | 沿每条 beam path 的 R80/R50、distal 80–20% | 给 median、95th percentile、max，而非只用中心轴 |
| Voxel dose | MAE、RMSE、bias、P95 absolute error | 分 >10%、>50%、>90% dose ROI；低剂量区用绝对差 |
| Clinical dose | D98/D95/D50/D2、V95、homogeneity/conformity；OAR Dmean/Dmax/Vx | 报绝对 Gy 和百分比差；不要仅用 DVH 视觉判断 |
| LETd | median/mean/P95、LETd-volume histogram、voxel MAE/P95 | 在 >10% dose 或足够 denominator mask 内；primary 与 all-hadron 分开 |
| 统计 | seed 间 SD/95% CI、combined uncertainty z-score | 区分 MC noise 与系统偏差；gamma pass rate 也给 CI |
| 物理审计 | interaction rate、species dose、energy balance、overflow | 尤其检查 distal tail 和 heterogeneity interfaces |

## 4. 建议预注册的接受标准

以下阈值适合做项目内部的起始 gate；最终应结合体素尺寸、TOPAS 统计不确定度和目标期刊调整，且必须在 hold-out CT 分析前锁定。

| 层级 | 建议 gate | 判定 |
|---|---|---|
| 水箱射程 | 全能量 R80 |Δ| ≤1 mm（粗体素可改为 ≤1 voxel） | 必须通过 |
| 水箱剂量 | 高剂量区 ≤2%；1D/2D gamma 2%/2 mm ≥95% | 必须通过 |
| 异质 phantom | R80 ≤2 mm；3D gamma 2%/2 mm ≥95%；界面无连续系统偏差 | 必须通过 |
| CT 3D dose | 每例 gamma 2%/2 mm ≥95%，3%/3 mm ≥98%，10% cutoff | 任何失败例单独解释 |
| Clinical metrics | target D95/D2 与主要 OAR Dmean/Dmax 多数 ≤2%，无 >3% 系统偏差 | 必须给 worst case |
| LETd | 高剂量 mask 内 median error ≤5%，P95 error 预设；峰/tail 单独评价 | 不能以 dose gamma 代替 |
| 稳定性 | production 参数进一步加严时关键量变化 <1%；seed CI 可接受 | 必须通过 |
| 守恒/队列 | overflow=0；energy closure 建议 <0.1% 或逐项解释 | 硬性 gate |

## 5. MAIGO 特有的必要消融与风险项

| 风险/近似 | 必做实验 | 论文中应如何表述 |
|---|---|---|
| 核反应由 TOPAS/Geant4 reaction/cascade package 驱动 | package 内能量 vs 插值/边界能量；独立 seed/package；分物种与 tail 验证 | “surrogate/reference-derived final-state model”，不可称独立 Geant4 replacement |
| 默认 CT material MCS 可能使用 all-water radiation length | `enable_ct_material_mcs` off/on，在骨、肺、空气界面比较 lateral spread 与 dose | 明确 production setting 与速度代价 |
| secondary straggling 默认可关闭 | off/on 比较 fragment tail、LET peak/tail 和 runtime | 报告为何选定最终设置 |
| neutral transport 可关闭 | off / first-interaction / full（若 package 可用）比较低剂量 tail、OAR dose、速度 | 将未输运中子/光子列为 scope limitation |
| 无完整电子/光子 secondary 与 decay chain | 选取对这些贡献敏感的低剂量 ROI，量化 TOPAS–GPU residual | 不把低剂量外周一致性过度外推 |
| cascade generation 有限 | generation 1/2/3/4 收敛；记录 residual local heat | 给出截断能量占比与对 dose/LET 的影响 |
| FP32 dose atomics（NVIDIA preset） | FP32 vs 可用高精度/CPU reference；顺序变化与大 histories 稳定性 | 量化数值误差而非假定可忽略 |
| `dose_output_scale` 等经验参数 | calibration/validation 严格分离；固定参数跨病例 | 同时报告 raw 与 calibrated dose match |

## 6. 性能 Benchmark 的公平报告方式

| 要素 | TOPAS 端 | MAIGO GPU 端 |
|---|---|---|
| 硬件 | CPU 型号、socket、physical cores、threads、RAM、NUMA | GPU 型号、显存、driver、功耗模式、oneAPI/SYCL/CUDA plugin |
| 运行设置 | threads、processes、physics list、cuts、scorers、I/O | profile、step/cutoff、queue capacity、FP32/FP64、scorers、I/O |
| 计时边界 | geometry/physics initialization、transport、scoring、write 分开 | JIT/warm-up、load/package、H2D/D2H、kernels、write 分开 |
| 重复次数 | ≥5 次，报告 median [IQR] | ≥5 次，首轮 warm-up 单独列出 |
| 工作量 | 相同 histories、particles、spots、CT grid、输出 scorer | 完全相同；不得为追求速度关闭用于 accuracy benchmark 的物理 |
| 输出 | histories/s、CPU core-hours、wall time | histories/s、wall time、peak memory；可加 energy/history |
| 加速比 | GPU wall time 对固定 TOPAS 并行配置 | 同时给单核、临床可用多核两个基准，避免 cherry-picking |

建议至少展示四行：单能水箱、SOBP、单野 CT、full-plan CT；每行同时给 `best/medium/fast` 的精度和速度。

## 7. 推荐论文图表清单

| 编号 | 主文/补充 | 图表内容 | 回答的审稿问题 |
|---|---|---|---|
| Fig. 1 | 主文 | MAIGO 输运与计分流程、TOPAS package 生成边界 | GPU 到底模拟了什么？ |
| Fig. 2 | 主文 | 100–400 MeV/u 水中 IDD、range error、横向 σ | 基础 EM/射程/MCS 是否正确？ |
| Fig. 3 | 主文 | 分物种 fragment dose/fluence + primary/all-hadron LETd | 核反应与生物相关量是否正确？ |
| Fig. 4 | 主文 | 肺/骨/空气异质 phantom dose、LET、差值 | 材料界面是否稳健？ |
| Fig. 5 | 主文 | 3 个代表 CT 的 dose/LET/gamma/DVH | 临床几何下是否匹配？ |
| Fig. 6 | 主文 | 全病例 gamma、clinical metrics、LET error 箱线/雨云图 | 结果是否跨病例一致？ |
| Fig. 7 | 主文 | accuracy–runtime–memory Pareto | 加速是否以可接受误差换取？ |
| Table 1 | 主文 | 软件、硬件、physics/scorer、输入参数 | 是否可复现？ |
| Table 2 | 主文 | 全病例定量结果与 worst case | 是否只展示最好病例？ |
| Fig. S1–S3 | 补充 | step/cutoff/history/seed 收敛 | 数值与统计是否稳定？ |
| Fig. S4 | 补充 | material MCS / straggling / neutral / cascade 消融 | 近似项贡献多大？ |
| Table S1 | 补充 | 每个 config、commit、seed、histories、overflow、energy balance | 能否审计和复现？ |

## 8. 发文前 Go / No-Go 检查

- [ ] calibration 与 hold-out validation 完全分离，最终病例没有逐例调参。
- [ ] TOPAS/GPU 的 beam、geometry、material、dose 与 LET 定义逐项核对并公开。
- [ ] 水箱、横向散射、碎片、LET、异质界面、SOBP、CT、性能八类证据齐全。
- [ ] 所有正式运行 queue overflow=0，能量闭合满足预设阈值。
- [ ] 至少 3–5 seeds，误差条包含 MC 统计不确定度。
- [ ] CT 不只给 gamma：同时给 dose difference、range、DVH、LET、worst-case 与失败区域。
- [ ] 代表病例不是手工挑最好结果；病例选择规则写入 Methods。
- [ ] raw 与 calibrated 结果同时保留；统一 scale 不在 validation set 上重新拟合。
- [ ] 速度比较使用相同物理/scorer/workload，并披露 TOPAS 并行硬件。
- [ ] 明确声明中性粒子、电子/光子、衰变、材料 MCS、有限 cascade 等限制。
- [ ] 所有脚本、配置、原始汇总 CSV、绘图代码和环境版本可复现。

## 9. 推荐执行顺序

1. **冻结协议与配置**：锁定版本、定义、阈值、calibration/validation 拆分。
2. **跑 A1–A5 水箱基础物理**：先排除 range、MCS、fragment、LET 的基础偏差。
3. **跑 A6、B2、B3 异质与消融**：定位 CT 中最可能的系统误差。
4. **跑 A7–A11 计划、收敛、统计和守恒**：确定 production 参数与 histories。
5. **冻结模型参数**：此后不再使用 validation CT 调参。
6. **跑全部 CT paired comparison**：先单野、再 full plan；自动生成全病例表。
7. **最后做 A12 性能**：使用已经通过 accuracy gate 的配置测速度。
8. **整理 worst-case 与 limitations**：失败结果也进入补充材料并给物理解释。

## 10. 建议输出目录结构

```text
validation_publication/
├── protocol/                 # 冻结阈值、版本、定义
├── manifests/                # 每次运行的 commit/config/hardware/seed
├── water_mono/
├── lateral_mcs/
├── fragments_let/
├── heterogeneity/
├── sobp_spots/
├── convergence/
├── ct_cases/
│   └── CASE_ID/
│       ├── topas/
│       ├── maigo/
│       ├── metrics/
│       └── figures/
├── performance/
├── summary_tables/
└── manuscript_figures/
```

---

### 最低结论边界

若以上验证通过，论文可稳妥声称：**MAIGO 在预先规定的碳离子能量、材料映射、束流与 CT 计划范围内，对 dose-to-medium 和规定定义的 LETd 与 TOPAS/Geant4 达到给定精度，同时获得相应 GPU 加速。** 不应仅凭高 gamma pass rate 声称 MAIGO 等价或替代完整 Geant4 physics list。
