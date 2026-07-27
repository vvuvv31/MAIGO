# Carbon LET_d 开发与验证记录

更新日期：2026-07-27

> **角色**：本仓库**当前明确下一步**的工作记录与验收清单。  
> 已实现总览见 [`README.md`](README.md)；其它非 LET 待办见 [`futureStep.md`](futureStep.md)。  
> 分支：默认在 `master` 上改（[`BRANCH_WORKFLOW.md`](BRANCH_WORKFLOW.md)）。

## 目标

在 MAIGO GPU 蒙特卡中实现与
[Villadslj/Topas-Extension](https://github.com/Villadslj/Topas-Extension)
`myHadronLET` 一致的剂量平均电子阻止本领 LET_d，并验证：

- 水中 primary C-12 LET_d；
- 水中 all-hadron LET_d；
- 单能束与 SOBP；
- 一维深度/横向曲线和三维体素结果；
- GPU LET scorer 的性能开销；
- 按粒子种类使用精确停止本领表的精度与速度差异。

## LET_d 定义

TOPAS `myHadronLET` 使用每一步的平均动能计算电子阻止本领：

```text
LET_d = Σ(Edep_local × electronic_dEdx / density)
        / Σ(Edep_local + E_delta)
```

其中：

- 分子使用局部沉积能量乘以电子阻止本领；
- 分母包含局部沉积以及该步产生的 delta electron 动能；
- 输出单位为 `MeV/mm/(g/cm3)`；
- primary C-12 与 all-hadron 分开累计。

GPU 保存分子、分母原始 moments，最终再相除，避免直接平均每一步 LET。

## 已完成的工作

### 1. 安装并验证 HadronLET TOPAS 扩展

- 在本机编译并安装了与当前 TOPAS 4.2.p3、Geant4 11.3.2
  匹配的扩展版本。
- 添加了 HadronLET、delta electron、停止本领、cascade 和弹性反应诊断
  scorer。
- TOPAS 最多使用 40 个线程，保留其他 CPU 资源。

相关文件：

```text
validation/topas/run_hadronlet_topas.sh
validation/topas/extensions/
validation/topas/HADRONLET_VALIDATION.md
```

### 2. GPU LET scorer

已实现：

- `scorerLET: true/false` YAML 开关；
- primary C-12 和 all-hadron 深度 LET_d；
- primary C-12 和 all-hadron 三维 MHD LET_d；
- C、B、Be、Li、He、H 等元素分组 LET_d；
- 可选的 p、d、t、He-3、He-4 独立 LET_d 诊断；
- FP64 分子/分母原子累计；
- LET scorer 关闭时不分配 LET buffers，也不执行 LET 原子加法。

可选同位素诊断：

```yaml
light_isotope_let_output_file: out/gpu_light_isotope_letd.csv
```

路径为空时不会产生额外的同位素 LET 原子累计开销。

### 3. 同版本停止本领和 delta electron 表

从 Geant4 11.3.2 的 `G4EmCalculator` 直接提取了 32 种 cascade
中出现的同位素停止本领，包括：

- H-1/2/3；
- He-3/4/6/8；
- Li、Be、B、C、N、O、F 的相关同位素。

YAML 开关：

```yaml
# 快速近似：使用 C-12 有效电荷缩放
use_particle_specific_stopping_power: false

# 精确模式：使用各同位素 Geant4 表
use_particle_specific_stopping_power: true
particle_stopping_power_file: data/ion_stopping_power_water_geant4_11_3_2.csv
```

缺失同位素会自动退回 C-12 有效电荷缩放。

表文件：

```text
data/stopping_power_water_geant4_11_3_2.csv
data/ion_stopping_power_water_geant4_11_3_2.csv
data/let_delta_electron_fraction_water_geant4_11_3_2.csv
```

生成脚本：

```text
validation/scripts/prepare_ion_stopping_power_tables.py
```

### 4. 定位轻碎片 LET 偏低的原因

初始精确同位素表仍出现：

| 同位素 | 初始 GPU/TOPAS LET 比值 |
|---|---:|
| proton | 0.812 |
| deuteron | 0.837 |
| triton | 0.969 |
| He-3 | 0.997 |
| He-4 | 0.875 |

TOPAS 与 GPU 的同位素 denominator 和组成已经基本一致：

- proton denominator 比值约 0.967；
- deuteron 约 1.002；
- triton 约 0.978；
- He-3 约 0.976；
- He-4 约 0.985。

因此问题不是轻碎片产额整体错误，而是剂量加权停止本领/低能能谱处理错误。

最终确认的主要原因是旧表只覆盖 `1–400 MeV/u`。GPU 对低于
`1 MeV/u` 的碎片始终使用 1 MeV/u 表值，丢失了 p、d、He-4
在 Bragg 高 LET 区域的贡献。

修复内容：

- 同位素和 C-12 表扩展到 `0.01–400.01 MeV/u`；
- 使用均匀 `0.1 MeV/u` 网格；
- GPU 对低于局部输运截止的剩余能量补计 LET moments；
- 默认碎片局部截止仍为 0.1 MeV。

修复后的比值：

| 同位素 | 修复前 | 低能表修复后 |
|---|---:|---:|
| proton | 0.812 | 0.938 |
| deuteron | 0.837 | 0.991 |
| triton | 0.969 | 1.153 |
| He-3 | 0.997 | 1.061 |
| He-4 | 0.875 | 0.979 |

Triton 和 He-3 占比较小，目前仍有产物能谱误差。没有为当前 SOBP
加入按 case 拟合的经验校准。

### 5. 已排除或量化的假设

以下因素不是当前 H/He LET 偏差的主因：

- C-12 delta electron 校准错误套用于 H/He：
  TOPAS 直接测得 H 的 delta 能量份额约 3.9%，He 约 6.0%，与 GPU
  相应能区表值接近。
- 核阻止本领：
  相对电子阻止本领通常低于约 0.1%，不足以解释 10–20% 偏差。
- charged fragment 弹性散射：
  单次事件与平均自由程诊断显示其连续能损贡献太小。
- secondary Bohr energy straggling：
  开启后没有改善 H/He LET。
- cascade 最大代数从 2 增至 4：
  只产生极小变化，代数截断不是主要原因。
- 专用 proton/deuteron/triton/He3/alpha 定义与 `G4IonTable`：
  `G4EmCalculator` 输出相同。
- 将碎片截止从 0.1 MeV 降到 0.01 MeV：
  精度基本不变，运行时间增加，因此未采用。

### 6. SOBP 验证结果

低能停止本领修复后，50–100 mm SOBP 的 all-hadron LET_d：

| 指标 | 结果 |
|---|---:|
| 平均相对偏差 | -0.227% |
| 中位绝对相对误差 | 0.785% |
| P95 绝对相对误差 | 2.163% |
| MAE | 0.572 MeV/mm/(g/cm3) |
| RMSE | 0.734 MeV/mm/(g/cm3) |

Primary C-12 在同一区域：

| 指标 | 结果 |
|---|---:|
| 平均相对偏差 | -0.193% |
| 中位绝对相对误差 | 0.734% |
| P95 绝对相对误差 | 2.005% |

指标文件：

```text
out/letd_sobp/particle_table_1M/exact/lowenergy_compare/metrics.json
```

### 7. 宽能量验证

100、200、300、400 MeV/u 的 all-hadron 中位绝对相对误差：

| 能量 | 旧同位素表 | 低能扩展表 |
|---|---:|---:|
| 100 MeV/u | 1.263% | 1.207% |
| 200 MeV/u | 4.270% | 3.719% |
| 300 MeV/u | 11.411% | 9.926% |
| 400 MeV/u | 7.838% | 7.245% |

说明低能表在所有能量上都有改善，但 300–400 MeV/u 的 fragment tail
仍受 cascade 产物能谱模型限制。

指标和图片：

```text
out/letd_energy_sweep/comparison_particle_tables_lowenergy_g4_11_3_2/
```

### 8. 性能

Titan RTX 上 1M SOBP：

| 配置 | 时间 |
|---|---:|
| 修复前、1 MeV/u 表下限 | 约 14.0 s |
| 0.01 MeV/u 低能扩展表 | 约 15.45 s |
| 低能表且截止降至 0.01 MeV | 约 16.66 s |

最终选择：

- 保留 0.01 MeV/u 表下限；
- 保留 0.1 MeV 碎片局部截止；
- 相对修复前约增加 10% 运行时间。

### 9. 1D 和 2D profile

使用修复后的 100k、3 mm 三维 SOBP 输出绘制：

```text
out/letd_sobp/3d_lowenergy/profiles/letd_1d_profiles.png
out/letd_sobp/3d_lowenergy/profiles/letd_2d_profiles_common_range.png
```

1D 图包含：

- 中心 `12 × 12 mm` 平均深度 LET_d；
- 58.5、73.5、88.5 mm 三个深度的中心带平均横向曲线。

2D 图包含：

- 73.5 mm 的 XY；
- 中心 XZ；
- GPU、TOPAS 和 GPU−TOPAS。

显示约束：

- GPU 与 TOPAS 使用同一个物理 LET range；
- 差值图使用同一个以 0 为中心的对称 range；
- 使用 GPU/TOPAS 1% 最大剂量区域的并集作为共同掩膜；
- 当前物理范围为 `0–240.74 MeV/mm/(g/cm3)`；
- 当前差值范围为 `±290.00 MeV/mm/(g/cm3)`。

绘图脚本：

```text
validation/scripts/plot_sobp_letd_profiles.py
```

### 10. 测试状态

最终构建及测试：

```text
3/3 tests passed
```

包含：

- `carbon_tests`；
- CT 重定向测试；
- dose mapping geometry 测试。

## 下一步要做什么

### 第一优先级：定位高能碎片 cascade 能谱

当前最主要的剩余误差是 300–400 MeV/u all-hadron fragment tail。
下一步不应该继续对 SOBP 曲线做经验缩放，而应直接比较粒子能谱。

需要给 GPU 和 TOPAS 输出：

- p、d、t、He-3、He-4；
- 粒子产生时的总动能和 MeV/u；
- 深度；
- 方向；
- 产生代数；
- 父粒子 Z/A；
- 父粒子发生反应时的入射能量；
- 区分 primary C-12 直接产物、第一代 cascade 和后续 cascade。

分别在 100、200、300、400 MeV/u 比较：

- 产额；
- 平均能量与能谱分布；
- 角度分布；
- 各代 denominator；
- 各代 LET_d numerator；
- 各代对 all-hadron LET_d 的贡献。

### 第二优先级：能量条件化的 cascade package

如果确认当前 GPU 产物谱偏硬或偏软，应将 cascade package 改为：

```text
projectile isotope × incident-energy bin × correlated final state
```

要求：

- 不独立缩放每个产物；
- 保留同一事件内粒子种类、能量和角度的相关性；
- 缩小高能区采样带宽；
- 稀有同位素不能用距离过远的事件填充固定窗口；
- 检查反应 Q 值、重残核能量和局部沉积的闭合；
- 用同一 Geant4 版本重新提取 package。

### 第三优先级：独立 case 泛化

任何 cascade 修复都至少需要在以下 case 上验证：

1. 当前 5–10 cm SOBP；
2. 100、200、300、400 MeV/u 单能束；
3. 新的 SOBP；
4. 至少一个 CT plan。

验收时同时报告：

- primary C-12 和 all-hadron LET_d；
- global/局部相对误差；
- 1D profile；
- 2D/3D profile；
- dose agreement；
- GPU 速度和显存变化。

### 第四优先级：提高三维参考统计量

当前三维图使用 100k histories，低剂量区仍有明显统计噪声。完成物理模型
修改后建议进行最终高统计量验证：

- GPU：1M；
- TOPAS：300k–500k，最多 40 CPU 线程；
- GPU/TOPAS 使用相同体素、坐标范围、显示范围和剂量掩膜。

高统计量只用于最终确认，不应用于生成 case-specific 校准参数。

## 当前结论

水中 SOBP 的 primary C-12 和 all-hadron LET_d 已经可以较好复现 TOPAS。
本轮最大的物理修复是补齐 `1 MeV/u` 以下的同位素停止本领和低能 LET
尾部，而不是按当前 case 乘经验系数。

剩余工作的重点已经从电子停止本领转移到高能核碎片的条件化产生能谱和
cascade 相关性。完成这一部分后，再进行新的 SOBP 和 CT case 泛化验证。
