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
-纵向/横向必须联合抽样（相关性保留），能量节点间插值另行定义测试，
  不得复用一维 quantile 插值。
