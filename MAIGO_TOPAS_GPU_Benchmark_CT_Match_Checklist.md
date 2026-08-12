# MAIGO 发文前 TOPAS/GPU Benchmark 与 CT Match 清单

> 适用范围：`vvuvv31/MAIGO` 的 `reconstruct` 分支；文档按 2026-08-11 工作树与当前源码校正。目标是验证 MAIGO 的 GPU condensed-history 碳离子输运在声明的能量、材料、几何和计分范围内与 TOPAS/Geant4 达到预注册精度，而不是宣称其等价于完整 Geant4 physics list。

### 清单状态与证据规则

- `未开始`：缺配对运行或主输出；`部分`：已有有效输出，但场景矩阵、seed、指标或 gate 不完整；`失败`：输出无效或触发硬 gate；`通过`：全部预注册条件满足。
- runner 打印 `DONE` 只表示输运结束，不等于通过。只有由冻结的原始输出生成、带 manifest 且通过自动 gate 的结果才能标为 `通过`。
- `benchmark/phantom/formal_*`、图片和 driver log 是当前本地证据；这些生成结果不由 Git 跟踪，dirty 工作树结果只能作开发证据，不能直接作发文快照。
- 历史文档或旧结果与当前源码冲突时，以 `TransportConfig::validate()`、当次 config、binary hash 和 manifest 为准。

## 0. 论文中必须先锁定的验证协议

| 项目 | 必须记录/固定的内容 | 建议写法或输出 |
|---|---|---|
| 代码版本 | MAIGO commit SHA、dirty/clean、source diff/patch、binary SHA-256、分支、编译器、oneAPI/SYCL、CUDA plugin、CMake preset/选项 | Supplementary Table S1；正式结果优先 clean tag/commit |
| 参考端版本 | TOPAS、Geant4、HadronLET extension 版本与 physics modules | TOPAS 4.2.p3 / Geant4 11.3.2（若实际如此） |
| 物理列表 | TOPAS physics modules/cuts；MAIGO `best/fast` 的精确开关、table/package 版本 | 发文 benchmark 只使用这两个 profile；配置与 package metadata 随稿公开 |
| 输入同源 | 粒子数、seed、能谱、spot 坐标/权重、束斑、发散、几何、CT grid、材料映射完全一致 | 自动 manifest 记录解析后参数和每个输入 SHA-256 |
| 剂量定义 | 两端均为 dose-to-medium；体素质量、归一化、单位转换一致 | 不允许为每个病例单独重标定 |
| LET 定义 | dose-averaged LET、纳入粒子种类、cutoff、restricted/unrestricted stopping power、单位与 denominator mask 一致 | 同时报告 primary-C12 与 all-hadron LETd；TOPAS extension 不输出原始 moments 时明确标 `N/A` |
| 随机统计 | 每个关键场景至少 5 个独立 seed；两端分别报告均值、SD/SE 和 95% CI | TOPAS/GPU 不要使用相同 RNG 序列作“配对”假设；先做各自重复性 |
| 后处理 | 相同 grid；若必须重采样，固定插值方法并同时保留原始 grid 结果 | 禁止只展示平滑后曲线 |
| 盲法阈值 | 在查看最终 CT 结果前锁定 gamma、range、dose、LET 接受标准 | protocol/README 中预注册 |
| 校准参数 | `dose_output_scale`、`straggling_scale` 等必须用独立 calibration set 确定 | validation 病例不可再次调参 |
| 队列完整性 | charged/cascade/neutral queue overflow 必须为 0；同时报告能量闭合 | 任一 overflow 非零则该运行不可作验证 |
| 运行完整性 | exit code=0、backend 确为 SYCL、histories 实际完成数与预期一致、所有必需 scorer 存在且非空/非全零 | 任一不满足则 fail closed，不进入指标计算 |
| 证据不可变性 | 原始输出只读归档；后处理脚本、参数、环境和输出 hash 入 manifest | 图表必须可由 manifest 指向的 raw data 一键重建 |

### 0.1 当前仓库的可执行入口

| 层级 | 当前入口 | 用途与边界 |
|---|---|---|
| A1–A12 矩阵 | `benchmark/phantom/A_README.md`、`A*_gpu.sh`、`A*_topas.sh` | 通过环境变量选 binary/device/input；支持无副作用 `--dry-run` |
| 静态预检 | `python3 benchmark/phantom/validate_A1_A12_contract.py` | 检查必需 config、A3/A9 矩阵、histories 上限和 3D scorer 体素尺寸；不替代运行验收 |
| 开发结果审计 | `benchmark/phantom/A1_A12_CHECKLIST_AUDIT_20260811.md` | 记录当前缺项；不是冻结的发文结论 |
| 指标摘要 | `benchmark/phantom/plot_A1_A12_checklist.py` | 目前只自动化部分 1D range/gamma、能量账本和 timing；不覆盖完整 A1–A12 gate |
| CT 工作流 | `ctplan.md`、`validation/scripts/`、`ctResult.md` | 几何/坐标、转换和当前证据边界；历史数值必须按当前 commit 重算 |

Runner 默认单次 GPU/TOPAS 不超过 100k histories，A7/A8 的 21 层权重精确合计 100k，不得用标量 histories override 变成每层 100k。若统计 gate 要求更高精度，优先聚合多个独立 ≤100k repeat；若确需放宽单次上限，必须先修订并冻结 protocol，不能运行后追认。

### 0.2 2026-08-12 A1/A5 开发诊断

- 新增 `benchmark/phantom/audit_bragg_peak_2pct.py` 固化 Bragg peak 2% gate：峰位置用 `abs(GPU peak depth - TOPAS peak depth) / TOPAS peak depth`，峰剂量用 TOPAS 峰中心 `+/-2 mm` 积分差。窗口积分用于避免 0.5 mm binning 与有限 histories 将相邻峰顶 bin 的统计交换误判为物理偏差；单 bin peak error 仍保留为诊断，不作为 range/peak gate。
- 当前 TOPAS 五 seed mean 对 GPU `best/fast` 共 30 个可比 dose/energy peak 全部通过：峰深度相对误差均为 `0.000%`，峰中心 `+/-2 mm` 积分误差范围为 `0.009--1.973%`。最接近阈值的是 400 MeV/u `best` A1/A4（`1.973%`）。完整冻结输出见 `benchmark/phantom/figures_bragg_peak_2pct_20260812/bragg_peak_2pct_audit.{json,md,png}`。
- A2/A6 的单 bin peak diagnostic 可达 `4.84--9.02%`，但相同 case 的峰深度误差为 `0%`、峰中心 `+/-2 mm` 积分误差为 `1.14--1.41%`，说明残差来自峰顶有限 bin/MCS 采样而非 range 或总 peak-region dose。不得据此重新拟合 stopping power 或 straggling。
- LET peak **位置**也已在 2% 内；部分 primary-C12 单 bin LET peak 高度不稳定，TOPAS 五 seed 95% CI half-width 在 300/400 MeV/u 与 A7 分别约为 `5.9%/24.1%/30.6%`，不能作为 2% 硬 gate。100 MeV/u all-hadron LET 高度仍有约 `18%` 的系统 fragment-mixture 残差，属于 A5 核碎片/primary-survival 问题，不是 Bragg peak range 问题，继续保留为未通过项。

- A1 四能量 TOPAS 五 seed mean 与 GPU absolute dose 的 Bragg peak depth 全部一致；峰值差为 `-0.19%` 到 `-1.42%`，`0--1.2 x peak` 积分差为 `-0.23%` 到 `+0.72%`。因此当前没有使用整体 dose scale 或修改 stopping-power 标定的依据。
- 旧 Bohr 方差遗漏了重粒子碰撞的 `Tmax/beta^2` 相对论项，历史非单调 `straggling_scale(E)` 实际在经验补偿该缺项。现已改为 condensed total-loss relativistic dispersion，四个单能配置共享单位 scale。五个 50k GPU seed 聚合后，400 MeV/u dose 峰值差从 `-2.77%` 改善到 `-1.10%`、distal 80--20% width 差从 `+0.073 mm` 改善到 `+0.003 mm`；300 MeV/u dose 峰值差从 `-1.54%` 改善到 `-0.80%`。
- A5 dose-Bragg-peak bin 的 GPU LETd 差约 `1.4--2.7%`。图中更大的 distal spikes 来自单次 50k GPU LET ratio 的低 denominator：200 MeV/u primary raw max `93.82%`，在 GPU denominator `>=1%` 本曲线最大值后 max 为 `2.24%`、P95 为 `0.27%`。
- GPU primary cutoff-tail LET moments 已与 secondary 路径对齐并加回归测试；该修复在 200 MeV/u 50k A/B 中把主峰差 `2.24% -> 2.21%`，说明主要剩余不确定度仍是 single-seed ratio statistics 和 all-hadron fragment composition，不应通过 clamp LET 值修正。
- 已用匹配旧正式 runtime 的 straggling 表完成 A5 GPU 五个 50k seed，20 个运行均为 generation 4、`secondary_queue_capacity=2.5M` 且 overflow=0；聚合时先求和 raw numerator/denominator 后相除。200/300/400 MeV/u all-hadron P95 error 从 single-seed 的 `7.68/8.67/9.72%` 降到 `5.58/5.56/6.50%`。100 MeV/u distal all-hadron 仍为系统性 fragment-tail outlier，需单独物理消融。
- 100 MeV/u species LET 诊断显示 26.25 mm 处 GPU primary-C12 LET 本身只差约 `3--4%`，主要偏差来自 all-hadron denominator 的 primary/fragment 权重（TOPAS primary fraction `70.37%`，旧 GPU `50.91%`），不是 B/He/H stopping-power 表的整体偏差。局部把 100 MeV/u scale 调到 `1.30` 虽可压低该点误差，却使相同 current-E/A 在不同入射束中使用不同参数并恶化 dose，已撤回。相对论共享模型将五 seed 该点 error 从 `-22.27%` 改善到 `-18.34%`，仍未通过；secondary straggling on/off 对该点仅改变约 `0.001` 个百分点。因此该残差必须从 nuclear fragment production/primary survival mixture 继续诊断，不得再用 straggling 拟合。
- 相对论共享模型的五 seed Bragg-peak-bin LET error 为 primary `0.26--2.50%`、all-hadron `0.80--2.43%`；20 个运行的 secondary/cascade overflow 均为 0。200 MeV/u step `0.1/0.05/0.025 mm` 相对最细点的 dose L1 为 `0.62%/0.45%/0`，Bragg +/-2 mm primary LET MAE 为 `0.91%/0.43%/0`，满足当前 0.5--1% 收敛判据但仍需 publication snapshot 复核。
- 五 seed dose ensemble 同时缩小高能统计差异：400 MeV/u `0--1.2 x peak` 积分差 `-2.05% -> +0.25%`，MAE/TOPAS peak `0.91% -> 0.42%`，P95 point error `7.74% -> 3.70%`。这是 dirty-worktree 开发证据；冻结 commit/binary hash 后必须重新生成发文快照。

## 1. TOPAS/GPU 必做物理 Benchmark（按优先级）

### A. 最小可发表集：必须完成

| 编号 | 场景与扫描矩阵 | 对比量 | 建议统计/接受标准 | 推荐展示 | 状态 |
|---|---|---|---|---|---|
| A1 | **均匀水中单能 pencil beam**：至少 100、200、300、400 MeV/u；建议覆盖模型最低/最高有效能量 | IDD、入口平台、Bragg peak、fragment tail、R80/R50、峰宽；横向 σ/FWHM 随深度 | R80 差 ≤1 mm（或 ≤1% range）；高剂量区局部剂量差 ≤2%；1D gamma 1 mm/2% 与 2 mm/2% 均报告 | 每个能量一张 IDD ratio panel；range error vs energy 汇总图 | 部分：已有四能量 IDD/range；横向与正式多 seed 待补 |
| A2 | **束斑与多重散射**：同一能量在入口、mid-range、peak 前后、tail 取横向 profile | lateral σ、FWHM、80–20% penumbra、halo | σ/FWHM 差 ≤1 mm 或 ≤5%；不能只比较中心轴 | 2D dose map + 4 个深度横向 profile + σ(z) | 部分：已有 200 MeV/u 3D scorer；横向指标和 difference map 待补 |
| A3 | **能损涨落/射程展宽**：monoenergetic 与真实 energy spread（至少 0、0.5%、1%） | peak width、distal 80–20%、R80 方差 | 各 seed 间差异在 CI 内；展宽趋势单调且与 TOPAS 一致 | distal fall-off 放大图 | 部分：已有 0/1%；0.5% 已配置未形成正式证据 |
| A4 | **核反应与碎片尾**：水中高统计单能束，至少 200 与 400 MeV/u | primary attenuation、核反应率、C/B/Be/Li/He/p 分物种剂量或 fluence、tail dose | 反应率和主碎片积分差建议 ≤5%；tail 单独评价，不能被全局 gamma 掩盖 | 分物种深度曲线、tail ratio、能量闭合表 | 部分：200 MeV/u 有分物种；400 MeV/u 分物种和反应率待补 |
| A5 | **LETd 水箱**：与 A1 同能量；primary-C12 和 all-hadron 分开 | LETd(z)、peak 位置/幅值、entrance/peak/tail ROI；GPU 原始 numerator/denominator，TOPAS moments 若 extension 不支持则记 `N/A` | 高剂量区 LETd median error 建议 ≤5%；峰值/尾部单列，不在低剂量噪声区用无限相对误差 | Dose+LET 联合图、LET difference/ratio、ROI 箱线图 | 部分：四能量 TOPAS/GPU 五 seed 开发比较已完成；primary cutoff-tail 已修复，100 MeV/u fragment-tail 与 clean publication snapshot 待补 |
| A6 | **异质 phantom**：当前 AABB air cavity/bone insert/offset bone；正入射与 10° 斜入射。如论文声称层状肺/软组织/骨，需另加 layered 配对输入 | 界面剂量、range shift、lateral spread、LETd、核碎片尾 | distal range ≤1–2 mm；界面前后分别做 gamma/差值；报告材料 MCS 开/关 | 材料条带 + depth dose/LET；界面局部放大 | 部分：已有 3 个正入射 IDD；3D/LET/斜入射待完成 |
| A7 | **SOBP / 多能量层**：临床宽度的 SOBP，先单轴再 2D 多 spot | plateau uniformity、proximal/distal edge、range、LETd 梯度 | plateau 均匀性差 ≤2%；3D gamma 2%/2 mm（10% cutoff）≥95%，并附 3%/3 mm | SOBP depth dose、LETd、gamma map/histogram | 部分：已有配对 1D dose/LET；有效 3D gamma 和多 spot 分析待补 |
| A8 | **spot plan 解析与批处理一致性**：TOPAS spot file 的 batched vs sequential；`TPS source` 是独立任意角 batched-only 路径，需单独已知答案测试 | 总剂量、逐 spot 权重、坐标变换、能量层、MHD/CSV 一致性 | 比较前检查实际总 histories；用多 seed 或统计容差，FP32 atomic 不要求 bitwise equality | 小型已知答案 spot pattern + difference map | 失败：现有 TOPAS 3D scorer 全零；已修输入，必须重跑后才能重评 |
| A9 | **数值收敛性**：`best` 扫 step = 0.1/0.05/建议 0.025 mm；`fast` 在 CT 上以 production step 为中心加严；另扫 relative energy loss、cutoff、cascade generation、queue capacity | R80、峰值、tail、LETd（仅 `best`）、运行时间 | 两个 profile 分开验收；选定 production 参数后继续加严变化 <0.5–1%；overflow=0 | accuracy–runtime Pareto 图 | 部分：已有的 0.2–1.0 mm 水箱扫描不属于正式 profile gate；需重建 `best/fast` 矩阵 |
| A10 | **统计收敛与重复性**：当前 protocol 为 10k/50k/100k，每点 ≥5 seeds；对独立 repeat 做 ensemble 收敛 | voxel uncertainty、R80/IDD/LET CI、gamma CI | 差异必须相对联合 MC 不确定度解释；以预注册 uncertainty 而非固定单次 10⁷ 作停止条件 | error vs effective total histories 的 log-log 图 | 部分：已有一组粒子数扫描；5-seed 正式矩阵待完成 |
| A11 | **能量守恒与审计量**：所有上述场景 | initial、deposited、escaped、beamline removed、untracked/residual、overflow energy | 闭合相对残差 `<1e-3`；所有队列 overflow=0。“可解释”只能作限制说明，不能把超阈值运行改判为通过 | 每类场景一行的 balance table | 部分：GPU log 可审计；A1–A10 汇总硬 gate 待完成 |
| A12 | **端到端性能**：warm-up 后同一 GPU；当前单次 10k/50k/100k，每点 ≥5 repeats；单 spot、SOBP、CT | wall/transport/kernel time、初始化/I/O、histories/s、voxels/s、真实峰值显存、能耗可选 | 报 median [IQR]；同时给 TOPAS CPU 核数/线程与硬件；显存 estimate 不得冒充实测 peak | throughput scaling、breakdown、accuracy–speed 表 | 部分：已有单水箱时间；5 repeats、SOBP/CT、I/O 拆分和 peak memory 待完成 |

### B. 强烈建议：显著增强审稿说服力

| 编号 | 场景 | 目的与关键输出 | 状态 |
|---|---|---|---|
| B1 | 不同 beam size/divergence 与 off-axis spot | 验证相空间和几何变换；展示 spot centroid、σx/σy、旋转/平移误差 | ☐ |
| B2 | CT 材料标定 phantom（air/lung/soft tissue/bone） | 分离 HU→材料/密度映射误差与输运误差；报告 WET 与 range shift | ☐ |
| B3 | 物理消融：material-MCS、secondary straggling、neutral transport 分别开/关 | 量化每项对 dose、LET、tail、速度的影响；对应 MAIGO 已有运行时开关 | ☐ |
| B4 | `best/fast` 两个发文 profile | 给出 accuracy–runtime–memory trade-off；`best` 强制 LET，`fast` 支持 CT 与解析 phantom dose-only，不支持 LET/minibeam | 部分：A0 与 A1–A12 fast/best 已跑，正式多 seed/显存实测待补 |
| B5 | reaction/cascade package 泛化 | package 内插能量与边界能量分开报告；训练/生成能量与验证能量严格拆分 | ☐ |
| B6 | GPU 可移植性 | 至少 NVIDIA 两代 GPU；若主张 SYCL 可移植，再加 Intel Arc/Level Zero，比较数值一致性 | ☐ |
| B7 | 重复构建/可复现性 | clean build、固定 seed、容器或环境锁文件、自动生成表图 | ☐ |

## 2. CT TOPAS/GPU Match：病例与实验设计

### 2.0 当前 CT 证据边界（不等于本清单已通过）

- `ctResult.md` 截至 2026-08-09 仅支持 RT06423、RT07575 和 20022516 三例 rotation-fixed、equal-history 的当前常规束 dose 结论；当前统一汇总只有 3%/0 mm，不是本清单要求的 3D 2%/2 mm gate。
- 这三例虽写出 LETd，但还没有使用统一 TOPAS-dose/denominator mask 完成当前代码的配对 LET 比较；旧 `out/fullplan_result` 和旧 fast/best 数值不得并入。
- RT06541 旧 TOPAS full-plan 参考已确认无效，重算与审计完成前必须排除。RT07575 minibeam 当前参考是带 sparse threshold 的 per-spot TOPAS-Dij，不得写成独立 whole-plan TOPAS scorer。
- 上述运行可用于发现问题和设计 protocol，但不能在补齐每例 manifest、多 seed、预注册 gamma/LET 指标前改标为 `通过`。

### 2.1 病例选择

| 分组 | 最低建议 | 选择原则 | 不可混用的角色 |
|---|---:|---|---|
| Calibration | 1–2 例或独立 phantom | 只用于固定 `dose_output_scale`、`straggling_scale`、HU/material 参数 | 不进入最终总体通过率 |
| Internal validation | ≥5 例；更理想 10–20 例 | 头颈、胸部/肺、盆腔或骨附近靶区；覆盖不同体型、射程和异质性 | 不再逐病例调参 |
| Stress cases | ≥3 个 | 空气腔、肺、厚骨、斜入射、field edge、小靶区、长射程 | 单独列出，不用平均值掩盖失败 |
| External/hold-out（若可行） | ≥1 中心或不同 CT protocol | 不同扫描仪/重建核/层厚/校准曲线 | 用于声明泛化能力 |

### 2.2 每例必须跑的配对条件

- [ ] 完全相同的 CT voxel grid、patient orientation、isocenter、gantry/couch angle、spot positions、energies、weights 和 histories。
- [ ] TOPAS 与 GPU 使用相同 HU→density/material 映射；记录 CCTG version、origin/spacing/dimensions/direction、density/section/`za_rel`/`I_eV` hash，另保存 material-label difference map。
- [ ] 在 transport 前用 `--plan-only` 或等价检查输出 source bounds、beam direction、isocenter 和入射面；TOPAS “转 CT”与 GPU “转束”必须通过已知坐标点/小 spot pattern 证明等价。
- [ ] 单野分别比较，再比较总计划；总计划不能替代 per-field 诊断。
- [ ] 分别运行 `best` dose+LET 和 `fast` dose-only；两者物理与 scorer 不同，性能差不得解释为纯 LET scorer overhead。
- [ ] 至少 5 个独立 seed；TOPAS 与 GPU 的统计不确定度分别估计。
- [ ] primary-C12 LETd 与 all-hadron LETd 分开；低剂量/低 denominator voxel 设预先定义的 mask。
- [ ] GPU 输出 raw deposited energy、dose Gy、LET numerator/denominator、物种剂量、队列/能量审计量；TOPAS 端只要 scorer/extension 可用就保存对应 raw moments，不可用反推值冒充原始输出。
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

### 3.4 指标实现必须冻结的细节

- Gamma 必须记录 reference/evaluation 顺序、global/local normalization、dose cutoff 的 reference ROI、插值方法、搜索步长/范围、边界处理、是否在 BODY 内计算以及实际评价 voxel 数。抽样 gamma 必须标明抽样数和固定 seed，不得与全体素 gamma 混报。
- Dose 转换必须在原始 grid 先用每 voxel 局部质量得到 dose-to-medium；不得先插值 deposited energy 再用统一质量除。任何质量保守 rebin 都必须实施 sum/integral 守恒检查。
- LETd 在两端各自由 moments 相除后再比较，不对 LETd 值本身做三线性重采样代替 moments 重采样。mask 至少同时锁定 reference dose threshold 和 LET denominator threshold。
- 所有 ratio/relative error 的分母必须是冻结的 TOPAS reference；低剂量和近零物种尾部使用绝对差或积分 ROI，不报无界相对误差。

## 4. 建议预注册的接受标准

以下阈值适合做项目内部的起始 gate；最终应结合体素尺寸、TOPAS 统计不确定度和目标期刊调整，且必须在 hold-out CT 分析前锁定。

| 层级 | 建议 gate | 判定 |
|---|---|---|
| 水箱射程 | 全能量 R80 `abs(Δ) ≤ max(1 mm, 1 voxel)`，且无随能量的单调系统偏差 | 必须通过 |
| 水箱剂量 | 高剂量区 ≤2%；1D/2D gamma 2%/2 mm ≥95% | 必须通过 |
| 异质 phantom | R80 ≤2 mm；3D gamma 2%/2 mm ≥95%；界面无连续系统偏差 | 必须通过 |
| CT 3D dose | 每例 gamma 2%/2 mm ≥95%，3%/3 mm ≥98%，10% cutoff | 任何失败例单独解释 |
| Clinical metrics | target D95/D2 与主要 OAR Dmean/Dmax 多数 ≤2%，无 >3% 系统偏差 | 必须给 worst case |
| LETd | 高剂量 mask 内 median error ≤5%，P95 error 预设；峰/tail 单独评价 | 不能以 dose gamma 代替 |
| 稳定性 | production 参数进一步加严时关键量变化 <1%；seed CI 可接受 | 必须通过 |
| 守恒/队列 | overflow=0；每个 production run 的 relative energy-closure residual `<0.1%` | 硬性 gate；超阈值运行可解释但不可改判 |

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

建议至少展示四行：单能水箱、SOBP、单野 CT、full-plan CT。水箱/SOBP 报 `best`；CT 报 `best` 与 `fast` accuracy–speed trade-off。

### 6.1 正式 run manifest 最小字段

| 分组 | 必需字段 |
|---|---|
| 标识 | case/run UUID、UTC 开始/结束时间、A1–A12/CT 场景标签、calibration/validation/stress 角色 |
| MAIGO | commit/branch/dirty；diff hash 或 patch；binary SHA-256；CMake preset/cache 和 compile definitions；backend/device/driver |
| TOPAS | executable SHA-256、TOPAS/Geant4/extension 版本、physics modules/cuts、threads、G4 data-set versions |
| 输入 | 实际 runtime config/TOPAS overlay、include 展开后参数、CT/spot/weight/table/package 路径与 SHA-256 |
| 运行 | seed、请求/完成 histories、spot/layer 分配、profile 与每个物理/scorer 开关、queue capacity |
| 完整性 | exit code、backend tag、S/C/N overflow 数与能量、能量账本各项与相对残差、必需输出 hash |
| 后处理 | script commit/hash、Python 与依赖版本、mask/gamma/rebin/归一化参数、指标 JSON 和图表 hash |
| 性能 | warm-up 策略、计时边界、wall/transport/kernel/I/O、实测 peak memory 的工具与采样间隔、功耗设置 |

Manifest 必须是机器可读文件（JSON/YAML），Markdown 摘要只是其派生视图。路径不能是唯一 provenance；每个关键输入和输出都要有内容 hash。

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
- [ ] 每个关键场景至少 5 seeds，误差条包含两端 MC 统计不确定度。
- [ ] CT 不只给 gamma：同时给 dose difference、range、DVH、LET、worst-case 与失败区域。
- [ ] 代表病例不是手工挑最好结果；病例选择规则写入 Methods。
- [ ] raw 与 calibrated 结果同时保留；统一 scale 不在 validation set 上重新拟合。
- [ ] 速度比较使用相同物理/scorer/workload，并披露 TOPAS 并行硬件。
- [ ] 明确声明中性粒子、电子/光子、衰变、材料 MCS、有限 cascade 等限制。
- [ ] 所有脚本、配置、原始汇总 CSV、绘图代码和环境版本可复现。
- [ ] A1–A12 每项都有机器可读的 pass/fail/partial 记录和缺项；没有用 transport `DONE` 或图片存在代替验收。
- [ ] 正式证据所需 runner、config、TOPAS input、指标脚本和 protocol 都已纳入受控版本，不依赖未跟踪工作树。

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
├── protocol/                 # 冻结阈值、版本、定义；含变更记录
├── manifests/                # 机器可读 manifest；一个 run 一个 UUID
├── raw/                      # 只读 TOPAS/MAIGO 原始输出，按 hash 审计
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
