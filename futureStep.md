# futureStep — 当前主线与后续计划

更新日期：2026-07-27

已实现能力见 [`README.md`](README.md)。  
分支约定：[`BRANCH_WORKFLOW.md`](BRANCH_WORKFLOW.md)（默认 `master`）。

---

# A. 当前主线：Carbon LET_d 与高能 cascade 能谱

## A.1 目标

在 MAIGO GPU 蒙特卡中实现与
[Villadslj/Topas-Extension](https://github.com/Villadslj/Topas-Extension)
`myHadronLET` 一致的剂量平均电子阻止本领 LET_d，并验证：

- 水中 primary C-12 LET_d；
- 水中 all-hadron LET_d；
- 单能束与 SOBP；
- 一维深度/横向曲线和三维体素结果；
- GPU LET scorer 的性能开销；
- 按粒子种类使用精确停止本领表的精度与速度差异。

## A.2 LET_d 定义

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

## A.3 已完成的工作

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

## A.4 下一步要做什么（主线优先级）

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

#### 已实现（2026-07-27）：GPU birth-spectrum scorer + 对照脚本

| 项 | 路径 / 开关 |
|----|-------------|
| YAML | `fragment_birth_spectrum_output_file: out/.../prefix`（非空即启用） |
| 配置示例 | `config/beam_200MeVu_birth_spectrum_smoke.yaml`、`config/beam_400MeVu_birth_spectrum_10k.yaml` |
| 输出套件 | `<prefix>_summary.csv`、`_mevu.csv`、`_depth.csv`、`_costheta.csv`、`_parent_mevu.csv`、`_parent_z.csv` |
| TOPAS 表→同格式 | `validation/scripts/prepare_topas_birth_spectrum.py` |
| GPU vs TOPAS 比较 | `validation/scripts/compare_fragment_birth_spectra.py`（`--generation N`） |

GPU 在 **queue fit 之前** 对所有产生的 p/d/t/He-3/He-4 做 atomic 直方图（产额×代数、MeV/u、深度、cosθ、父粒子 MeV/u 与 Z），避免队列溢出偏置谱。

#### 400 MeV/u 初步结论（10k GPU，G4 11.3.2 package）

| 对比 | 结果 |
|------|------|
| GPU gen0 vs TOPAS primary package | 产额比 ≈ 0.95–1.01，mean KE 比 ≈ 0.99–1.01 → **初级末态采样健康** |
| GPU gen1 vs TOPAS cascade package（全包） | 产额比 ≈ 0.35–0.47；He-4 mean KE 比 ≈ **0.31** — 见下方条件化说明 |

#### 已实现（2026-07-27 续）：多能量套件 + 能量条件化 cascade 采样

| 项 | 路径 |
|----|------|
| 多能量 runner | `validation/scripts/run_birth_spectrum_energy_suite.py` |
| 父粒子能量分析 | `validation/scripts/analyze_cascade_parent_energy.py` |
| 结果目录 | `validation/results/birth_spectrum_energy_suite/` |
| 采样改动 | `select_cascade_interaction_energy_conditioned`：按 **相对能量带宽** 选事件，不再用固定 8 事件 index 窗口；`cascade_event_energy_scale` clamp 到 `[0.25, 4]` |

**100/200/300/400 MeV/u 套件（10k，seed 20260727）要点**：

| E | gen0 产额/KE 比（相对 **400 MeV primary package**） | gen1 说明 |
|---|------------------------------------------------------|-----------|
| 100–300 | 明显 <1，随 E 升高趋近 1 | **预期**：参考包来自 400 MeV TOPAS 运行；低能束核反应更少、产物更软，不宜当作 gen0 失败 |
| 400 | gen0 ≈ **1.00** | gen0 与同版本 primary package 闭合 |

gen1 与 **完整 cascade package** 的粗比不能直接定罪采样：

- cascade package 含全部后续反应代数与全部弹核种类；GPU gen1 只是第一代级联；
- `_parent_mevu` 直方图未分 generation，混有 gen0（C-12@高能）与 gen1（碎片@降能）；
- package He-4 对父粒子能量条件极强：父 10–50 MeV/u 时 mean MeV/u ≈ 2–9，父 350–400 时 ≈ 190–240。

#### 已实现（2026-07-27 再续）：generation 分箱 + 条件化 gen1

- 直方图布局：`species × generation × bin`（mevu/depth/cos/parent_mevu/parent_z）  
- 联合谱：`_parent_product_mevu.csv`（parent MeV/u × product MeV/u）  
- TOPAS 过滤：`prepare_topas_birth_spectrum.py` 支持 projectile Z/A、parent MeV/u  
- 条件化 runner：`run_gen1_conditioned_compare.py`、`compare_birth_joint_parent_product.py`  
- 报告：`validation/results/birth_spectrum_gen1_conditioned_report.md`

**关键结论（400 MeV，10k）**：

- gen1 He4 父核以 **Z=1/2（~76%）** 为主，C-12 仅 ~7%  
- 相对 **全 package** 的 He4 “过软” 主要是父核混合偏差（package 高能 bin 由 C12 碎裂主导）  
- 按父核切开后：vs Z≤2 residual **GPU 偏硬**；vs C12 碎裂 **GPU 偏软**  
- gen1 **proton** 按父能量 bin 的 mean MeV/u 与 package **≈0.85–1.05**，采样大体健康  

#### 已实现（2026-07-27）：LET 精度 — secondary 队列容量

| 项 | 说明 |
|----|------|
| 根因 | CUDA auto `secondary_queue≈4N` 在 400 MeV/100k 下 **cascade 溢出 ~3.7e5** |
| 修复 | LET 配置/脚本 `≥ max(2.5e6, 25N)`；cascade 能带优先 ≤15%；scale `[0.85,1.18]` |
| all-hadron 中位\|rel\| | **300: 9.9%→7.0%；400: 7.2%→4.1%**；峰后 GPU/TOPAS **0.70→0.88** |
| SOBP 50–100 mm | 仍约 **0.8%** |
| 报告 | `validation/results/letd_accuracy_improvement_2026-07-27.md` |

#### 已实现（2026-07-27 续）：远端碎片产生深度与 cascade 代数

重新检查 400 MeV/u 时发现，总产额和总能谱闭合会掩盖空间分布误差。
修复 `prepare_topas_birth_spectrum.py` 后，可从 interaction/reaction 表把
TOPAS 产物正确回填到产生深度；排除 cascade 表中重复的 source-track
primary C-12 反应后，GPU/TOPAS 的总产额仍约为 0.97–0.99，但
320–360 mm 的远端产生率为：

| 粒子 | GPU/TOPAS 产生率 |
|---|---:|
| proton | 0.948 |
| deuteron | 0.906 |
| He-3 | 0.833 |
| He-4 | 0.878 |

这与相同深度区间 all-hadron LET_d 偏低同向。大队列下把 cascade
上限从 2 代提高到 4 代后：

| 能量 | 2 代 median \|rel\| | 4 代 median \|rel\| |
|---|---:|---:|
| 300 MeV/u | 7.044% | 6.949% |
| 400 MeV/u | 4.115% | 4.025% |

400 MeV/u P95 从 27.03% 降到 24.73%，cascade interaction 数从
128924 增至 133524，更接近 TOPAS 去除 primary 后的 135271。
运行时间由 16.59 s 变为 17.47 s（约 +5.3%）。因此 LET 生产配置
现在默认使用：

```yaml
maximum_cascade_generations: 4
secondary_queue_capacity: 2500000  # 100k LET validation
use_particle_specific_stopping_power: true
```

SOBP 100k 泛化检查没有退化：50–100 mm all-hadron median |rel|
为 0.818%，mean bias 为 −0.166%。

同时排除了两个候选项：

- 开启 secondary Bohr straggling 未稳定改善宽能量 LET（400 MeV/u
  median 4.115%→4.137%），继续保持关闭；
- 新增 `IonNuclearLETNtuple` 后确认，TOPAS 非弹性 step 的 LET
  权重只使用该 step 的局域电离沉积，不是母粒子的全部剩余能量；
  这些 step 仅占 400 MeV/u 全局 LET numerator 约 0.032%，不能用
  GPU nuclear residual 直接补 LET。

**仍待做**：

1. cascade **截面/反应率** 按弹核 vs TOPAS（总率已闭合到约 1.3%，
   仍需逐 Z/A）
2. 各代 LET numerator/denominator
3. 320–400 mm 远端 transported-parent population 与产生率闭合；
   当前高能 all-hadron 仍有约 4–7% 中位残差

#### 已实现（2026-07-27）：parent Z/A × energy × depth package A/B

- v3 binary 为每个 interaction 保存 reference depth；
- 数据实际重排为
  `projectile Z/A × 2 MeV/u × 10 mm depth × exact energy`；
- GPU 通过二分定位二维 cell，不拆散 correlated final state；
- v1/v2 向后兼容；
- YAML 开关：

```yaml
cascade_condition_on_reference_depth: false
```

A/B 结果表明 absolute depth 含有 400 MeV/u 水箱 case 信息：

| 模式 | 300 MeV/u median \|rel\| | 400 MeV/u median \|rel\| |
|---|---:|---:|
| v2 energy-only | 6.949% | 4.025% |
| v3 energy-bin-only | 6.865% | 4.139% |
| v3 energy × absolute depth | 7.195% | 3.968% |

因此 depth 模式虽然对同源 400 MeV/u 略有改善，但在实际更重要的
300 MeV/u 上退化，判定为过拟合风险，保留为诊断开关而不设为默认。
SOBP 的 v3 energy-only median 为 0.852%（v2 为 0.818%），差异很小。

完整报告：
`validation/results/letd_conditioned_cascade_2026-07-27.md`。

### 第二优先级：能量条件化的 cascade package

**运行时采样已部分完成**（紧能带 + 窄 scale；生产 LET 大队列）。  
若确认仍需改数据布局，则将 cascade package 改为：

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

## A.5 当前结论

水中 SOBP 的 primary C-12 和 all-hadron LET_d 已经可以较好复现 TOPAS。
本轮最大的物理修复是补齐 `1 MeV/u` 以下的同位素停止本领和低能 LET
尾部，而不是按当前 case 乘经验系数。

剩余工作的重点已经从电子停止本领转移到高能核碎片的条件化产生能谱和
cascade 相关性。完成这一部分后，再进行新的 SOBP 和 CT case 泛化验证。

---

# B. LET 主线之后：近期项

## B.1 中性粒子正式闭合

| 状态 | 说明 |
|------|------|
| 已有 | 方案 D / full 模式、neutral package 加载、interim `neutral_local_kerma_fraction` |
| 待做 | 用正式 neutron/gamma package 替换 interim local kerma |
| 待做 | 水中与 TOPAS neutral-origin 剂量闭合；再评估 CT 上是否可开 `enable_neutral_transport` |
| 注意 | CT 上 11.1.3 water-derived neutral package 曾恶化 IDD correlation；患者材料需单独验证 |

## B.2 高能峰高残差（300/400 MeV/u）

- 带电 IDD 峰高仍可能有约 −3–4% 量级残差  
- 在 cascade 能谱修复（A.4）后复测，避免在谱错误时对 SP/straggling 经验调参  
- 检查限制能损语义与碎片 cutoff 是否仍主导峰区展宽

## B.3 材料相关 reaction / cascade final-state

| 状态 | 说明 |
|------|------|
| 已有 | 水包 + 材料相关 XS / mass-SP；骨/肺 **诊断** package（G4 11.3.2）已导出 |
| 待做 | 与 production 参考 **同 Geant4 版本**（目标 11.1.3）的骨/肺/组织 final-state 包 |
| 待做 | CT 按 material_id 选择 package；Q 值 / 重残核 / 局部沉积闭合 |
| 禁止 | 用 11.3.2 诊断库静默替换 11.1.3 production 而不做对波 |

## B.4 CT / TOPAS 对照补强

- 等权层、多 spot 在**统一旋转修复**后复算（历史 prelim 层需刷新）  
- 提高 TOPAS prelim 统计（≥2e5–5e5）再评 local γ 噪声底  
- 全 plan TOPAS MC 参考（在中性/材料包满意后）  
- CT binary 0.25 mm origin 元数据债务：下次从 DICOM 完整重建时统一，禁止只改单边配置  
- 跨病例盲测：固定 MU↔ions 响应因子后测新计划，避免单病例过拟合

---

# C. 中期（计划接口与泛化）

## C.1 多野 / 任意角度计划

| 已有 | 待做 |
|------|------|
| TPS 90° 轴置换 + TOPAS spots 批处理 | 通用 `beam_dir` / gantry 循环脚本叠加多野剂量 |
| opt-in `tpsSource`（碳离子 + PBS CSV） | 多粒子种、更完整 couch/collimator/患者方位（HFS/HFP…） |
| 单计划 batch primary | 外层脚本多角度 seed 流与绝对 MeV/primary（或 fluence）叠加 |

最小增量曾规划（仍可参考）：

1. `TransportConfig` 显式 `beam_dir_*` + `isocenter_mm`（与现有 tps_source 协调，勿双轨）  
2. 先支持与体轴共线 ±z smoke  
3. `run_ct_plan_angles.py`：两角度 vs TOPAS 两野  

## C.2 验证与论文资产（暂缓优先）

- 论文级总图与消融表（用户曾要求暂缓）  
- 固定 seed 套件的一键回归报告（水 + hetero + CT smoke + LET）  
- 跨能量 / 跨几何 acceptance 清单写入 CI 或脚本门禁

---

# D. 性能与工程（以后做）

以下**不改变**已锁定物理约定时再做；每步独立 A/B，对照当前 full-physics 基线。

## D.1 次级输运

| 项 | 说明 |
|----|------|
| **P4 step-level wavefront** | 每 worker 固定 quantum steps 后 compact；缓解 secondary 长尾 SIMD 空等 |
| secondary 调度参数 | `secondary_batch_size`、local_size（CUDA 128 / L0 256）统一 A/B |
| CUDA secondary 非进度尾 | 历史 TITAN RTX oneAPI-CUDA 路径 secondary 极慢；需同 commit、同 FP64、同 seed 严格 A/B（见 archive 中 cudaVSlevel0 笔记） |

## D.2 Kernel / tally

| 项 | 说明 |
|----|------|
| **P5** 编译期 kernel 特化 | CT on、neutral off 等恒定分支消掉；防寄存器爆炸 |
| **P6** local/tiled tally | 热点 voxel 本地合并再 flush FP64；患者 CT 需测 hit rate |
| **P7** 可选 FP32 batch tally + FP64 reduction | 仅 fast mode；必须过 gamma / 能量闭合 |
| FP32 vs FP64 dose atomic 统一 A/B | Level Zero 生产多为 FP64；历史 CUDA 曾用 FP32，不可混比性能 |

## D.3 分级物理 / 快速剂量（仅 preview）

生产默认保持 `energy_cutoff_MeV=0.1`、`maximum_relative_energy_loss=0.005`、full cascade。

| 档位 | 用途 |
|------|------|
| `preview-primary` | 坐标 / 射程 smoke |
| `direct-secondary` | 直接带电次级、无 cascade |
| `full-cascade` | 最终报告默认 |

另测 `energy_cutoff_MeV=1.0` 等 **fast-dose** 配置：必须过 3D γ、R80、积分、峰位、能量闭合后才能标为 preview，**不得**静默替代最终剂量。  
每个输出记录 backend / physics-tier 与关闭的能量通道。

## D.4 架构债（低优先级）

- 拆分 `transport_sycl.cpp` 巨型单体（几何 / primary / secondary / neutral / scorer）  
- 收窄 `TransportConfig` 职责或分组  
- Serial 与 SYCL 能力不对齐：完整路径依赖 SYCL + TOPAS，无全功能 CPU oracle  

细节见 [`structure.md`](structure.md) §14。

---

# E. 明确不做 / 已排除的方向

- 为贴 TOPAS 或 `physical_dose` 做**按能量/按病例**的 SP、straggling、MCS 经验 scale  
- 把 ridge/能层拟合权重当作跨病例通用物理（入口面 bug 修复后已废弃该路径作 production）  
- 仅为性能用 2 mm 下采样 CT 替换细网格生产（实测端到端加速可忽略，次级仍主导）  
- 在 LET 主线未完成前对 SOBP LET 曲线做 case-specific 乘子校准  

---

# F. 建议验收门槛（任意新改动）

- 构建 + `carbon_tests` 通过  
- queue overflow = 0  
- energy-balance 不劣于当前统计可解释范围  
- 固定 seed 对照：IDD / R80 / 积分 / 3D dose / 峰位  
- 报告 2%/2 mm 或 3%/3 mm γ（视场景）  
- 同时报 wall time、kernel time、histories/s、steps/history，避免把“少算物理”写成“GPU 加速”  

---

# G. 历史文档位置

下列工作笔记内容已并入 README / 本文件，原文移至 `docs/archive/` 备查：

- `codex.md` — TPS 90° CT 性能 P0–P8  
- `grok.md` — CT gamma、入口面修复、full-plan 标定  
- `furtherStep.md` / `nextStep.md` — 中性、emittance、异质、CT 执行记录  
- `compileAndRun.md` — 构建命令（已写入 README）  
- `cudaVSlevel0.md` — CUDA vs Level Zero 性能分析  
- `GPU_MC_TPS_Source_Module_Design.md` — TPS source 设计（实现契约已写入 README）  
- `intel_oneapi_carbon_ion_gpu_monte_carlo_guide.md` — 长篇从零开发指南与早期状态  
- 原根目录 `LET.md` — 已并入本文件 A 节  
