# 能量分配契约（Step 05）：防止电子效应双算

> 状态：契约成立是改 GPU 的前置条件。本文件只定义能量归属与互斥规则，
> 不授权任何 GPU 修改。任一问题无答案即停在 Step 05。

## 0. 现实现状（代码证据，非变量名推断）

- GPU stopping 表 `data/schneider/schneider_stopping_v1.bin` 由
  `G4EmCalculator::ComputeTotalDEDX`（非受限总量，含 δ 射线带走能量）提取，
  证据：`/home/wuwei/topas/extensions/CarbonSchneiderStoppingPowerDump.cc:187`
  `em_calc.ComputeTotalDEDX(total_energy, projectile, mat)`。
  结论：**GPU 的 `mean_loss_MeV`（`src/transport_sycl.cpp:2948`）对应总电子
  能损，不是受限 stopping。**
- 本步沉积 `deposited_MeV`（`:2950`）= `min(mean_loss, energy)`，再经**一次**
  straggling 抽样（`:2952–3011`，RNG tags 0/1/2），即
  `local_voxel_deposit_MeV` 的起点是**已 straggle 的总电子能损**。
- v1 横向重分配（`:3016–3031`，RNG tags 17/18）：
  `moved_MeV = deposited_MeV × clamp(moved_fraction, 0, 0.5)`，
  终点 = 源 + radius × 横向（`:3042–3044`，`rotate_local_direction(...,0)`，
  纵向分量恒为 0）。
- v1 `moved_fraction` 的分母是 **scorer 内沉积剂量**，不是电子出生能量：
  `tools/compile_schneider_delta_tail.py:34`，
  `moved = outside_2.01mm_dose / total_lateral_dose`（100–200 mm 求和区间），
  数值 150/200/225 MeV/u → 8.97%/10.24%/10.69%。
- 生产阈：诊断与 v1 表均为 TOPAS 默认 `SetDefaultCutValue(0.05*mm)`，
  证据：`/home/wuwei/topas/OpenTOPAS-4.2.3/physics/TsModularPhysicsList.cc:433`，
  运行配置未覆盖。阈下能量以 C12 受限损失留在局部。

## 1. 实测五项能量（v2 配对，200 MeV/u、HU=-1000、均匀 slab、EM-only，seed 917001）

`/mnt/sda/wuwei/electron_v2_validation/v2_new/analysis_slab.json`：

| 项 | MeV | 说明 |
|---|---|---|
| 父 C12 电子过程总损失 (incident − primary_outgoing) | 424.1054 | 非受限总量 |
| 父 C12 局部沉积 primary_local（含阈下受限损失） | 276.5991 | 留在局部 |
| 显式电子根出生能量 birth | 147.5063 | 阈上 δ |
| 电子家族沉积 | 144.1332 | slab 内 |
| 电子家族逃逸 | 3.3731 | 离开 slab |

闭合：`primary_local + birth = 424.1054 = parent loss`（~1e-4 精确）；
`birth = deposit + escape`（残差 3.3e-15）。
关键比例：**birth/parent-loss = 34.78%**，而 v1 只搬走 ~10.24%。
差值（~24.5% 的 parent loss）是 v1 有意留在局部的近端 δ，不欠账。

## 2. 八问答复

1. **新模型从哪个现有能量项取能量？**
   与 v1 相同：本步已 straggle 的 `deposited_MeV`（总电子能损）。
   不得另立电子产生子程序从慢化能量中二次取能。
2. **可迁移比例分母是什么？**
   必须是**本步 `deposited_MeV`（总电子能损）**，且分子分母配套重标定。
   禁止把“电子家族沉积中 28.9% 前移超 0.5 mm”（分母是家族沉积）直接当成
   “全部 stopping 的 28.9%”（分母是总能损）。两者分母不同（§1：34.78% 关系）。
3. **原局部沉积减去多少？**  `local -= moved_MeV`，精确到原子加法所记账面值。
4. **新位置增加多少？** 同值 `moved_MeV` 原子加到终点体素；出 scorer 记
   `device[2]` 逃逸；跨材料终点按现有 fallback 留局部并单列计数（`:3106–3110`）。
5. **scorer 外能量如何记账？** 沿用现有三类账：
   moved_inside / cross_material_fallback_local / escaped_3d_scorer，
   全 shard 零 overflow 仍是硬门禁（溢出即拆分重跑）。
6. **与旧横向模型是否互斥？** **互斥，强制。** 同一步能量只能走 v1 或 joint
   其中之一。joint 若上线，必须整体替代 v1（独立验证入口，Step 09），
   禁止“v1 保留再叠加 34%”、禁止把 `moved_fraction` 直接改成 0.34。
7. **与显式 electron transport 是否互斥？** **互斥。** GPU 无显式电子输运；
   任何未来显式输运上线时，本重分配对同一能量必须关闭（双算）。
8. **是否重复施加 straggling？** **不重复。** straggling 只在 `:2952–3011`
   施加一次；重分配只搬运 `deposited_MeV`（tags 17/18 只做查找与方位角，
   不重抽样能损）。joint 实现必须复用同一 `deposited_MeV`，不得另起
   straggling 抽样。

## 3. 给 Step 08/10 的约束

- 联合核的分母若选“阈上出生能量”，则必须先乘 `birth/parent-loss` 转换到
  `deposited_MeV` 基准（200 MeV/u、HU=-1000 处实测 0.3478，随能量/密度变化，
  不得当常数），再与 v1 的 0.1024 对账：joint 搬走总量 − v1 已搬量 = 净新增，
  且净新增必须全部落在可解释的纵向迁移内。
- 纵向/横向必须联合抽样（相关性保留），能量节点间插值另行定义测试，
  不得复用一维 quantile 插值。

## 4. 附录 A：纵向补充通道（2026-09-05，section-0 forward kernel）

> **2026-09-06 撤销验收声明：以下保留为历史实现记录，不是当前执行规范。**
> 逐病例 scale 不是独立验证；RT06423 G30 下降约 0.028pp，不是持平。
> “越界误差 <10%”、源密度缩放适用任意路径、守恒即可证明无双算均未得到独立验证。
> 当前实现只允许 smoke/scale=1、150–225 MeV/u、参考密度
> 0.01131606474518776 g/cm³ 的均匀 section-0；域外不搬运并计数。
> 路径按体素面精确分段，遇其他密度/材料余量留源（未验收近似），出网格记逃逸。
> 未启用候选时恢复既有 tally 类型，不把精度切换当作物理修复。
> 候选始终不获质量验收。200 MeV/u 开关均有共享 writer 失败，后续需先定位；
> 独立密度/能量/界面/联合响应验证仍 BLOCKED，不以旧全量 Gamma 替代。

本附录记录 v1 横向 LUT 之外新增的纵向（前向）补充通道的设计偏离与证据。
v1 文件、`moved_fraction` 数值与横向量化表均未改动。

- 动机：三病例 Global 1%/1mm 失败集中在入射端 section-0 空气（RT06423
  86.7%、RT07575 89.8%、20022516 64.3%，GPU 偏高），且偏置随入射面距离
  衰减（~15–20 mm 尺度），均匀截面内幅度均匀——不是束宽/发散问题。
  HU=-1000 slab 横向积分（= 宽束深度剂量形状）显示 TOPAS 在 2–100 mm
  以 deficit(z) ≈ A·exp(−z/λ)（A ≈ 6.7–7.5%、λ ≈ 17–20 mm）低于 GPU 平台，
  即 GPU 缺少前向电子输运。纯横向终点（纵向分量恒 0）无法修正纵向分布。
- 能量分母：与 v1 一致，取本步已 straggle 的 `deposited_MeV`；
  未使用电子家族能量、28.9% 或 0.34 等其它分母。LUT 的 A 由剂量 deficit
  拟合（分母即局部沉积剂量），与 v1 分母口径一致。
- 与 v1 的关系（对 §2-Q6 的澄清）：两通道作用于正交投影——横向搬运保持
  深度分层不变（slab 横向尾部已验证），纵向搬运保持横向积分不变；
  slab 上横向-only 残留 +6.4% 入射 deficit，纵向补上后残留 ±0.5%，
  证明二者互补而非双算。能量守恒是逐项精确的（下）。
- 跨材料终点（对 §2-Q4 的澄清）：纵向采用沿程均匀分布（连续慢化图像），
  行经体素（空气或组织）按路径份额原子加分；出 scorer 段走逃逸 sink；
  scorer 内但 CT 网格外段留在产生体素并单列计数。点沉积曾在空气-组织
  界面处造成堆积（单 shard p99 +1.15% Dmax），分布形式消除该问题。
  横向通道的 fallback 语义保持不变。
- 相关性（对 §3 joint 约束的说明）：横向（RNG tags 17/18）与纵向（tag 19）
  独立抽样，未保留真实 δ 的纵-横关联；slab 横向积分验证对该近似不敏感
  （±0.5%），记为已知近似。
- 精度门禁：分布搬运产生大量亚 ulp(float)份额（slab 通道体素量级），
  FP32 原子加会静默丢失（已用模拟 + 50k/500k 缩放试验证明），触发能量闭合
  门。因此纵向通道在 FP32 构建拒绝运行（fail-closed，
  `CARBON_DOSE_FP32` 下 plan-only 与 transport 均抛错）；深度一维 tally
  （`dose_device`、`in_fov_dose_device`) 常驻 FP64（`DepthAtomicT`），
  体素三维数组保持 FP32 快路径。FP64 slab 全闭合 `voxel/ingrid = 1.0`。
- 距离的密度标度：电子 CSDA 射程 ~1/ρ；LUT 在 slab 空气
  （ρ_ref = 0.01132 g/cm³）标定，病人 section-0 空气约 0.011–0.06。
  采样均值按 ρ_ref/ρ_source 缩放（钳位 [0.15, 4.0]），均匀 slab 下恒为 1
  （slab 验证不受影响）。该标度使 RT07575 深层致密空气的过修正减半，
  面元残留基本不动。
- LUT 口径：`schneider_section0_c12_delta_longitudinal_v1.csv`
  存 NOMINAL 值（物理拟合 × kernel 映射 A×1.2、λ×1.3；均匀分布核的等效
  deficit 为 0.83A、0.76λ，200 MeV/u slab 实测），`scale` 旋钮默认 1.0
  （slab 真值），病人空气操作点另行标定（病人空气密度约 10× slab）。
- 三病例全量验证（FP64 构建，各 20/20 shard 质量门全过）：
  RT06423 s=0.63：G11 98.58→99.66、L11 87.50→91.72、G30 99.968→99.940、
  L30 92.29→95.08，面元残留 ±0.15%；
  RT07575 s=0.75：G11 96.74→98.16、L11 83.32→85.78、G30 99.28→99.49、
  L30 89.14→91.08，面元 ±0.12（深层 −0.41）；
  20022516 s=0.90：G11 98.79→99.17、L11 84.97→86.10、G30 94.27→94.39、
  L30 81.36→83.23，面元 ±0.19。12 项 Gamma 11 升 1 平（−0.03%）。
  全量剂量见 `/mnt/sda/wuwei/schneider_longitudinal_final/`。
- 能量覆盖：纵向查找越界取邻边值（clamp），不回零。Slab 拟合显示弱能量
  依赖（150→225 仅 +12%），越界 clamp 误差 <10%；而回零会丢掉整束的部分
  能量（RT07575 有 40.2% 权重 >225 MeV/u，20022516 有 30.6% <150 MeV/u），
  属 all-or-nothing 错误。横向表保持越界回零（v1 基线不动）。
- 本附录不改变 v1 横向表、MCS、package 与全局剂量 scale；20022516 肺内
  冷热交错另案处理（本通道打开前后其形态不变，待查）。
