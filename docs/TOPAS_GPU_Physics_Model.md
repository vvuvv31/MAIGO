# TOPAS/Geant4 与 GPU 碳离子蒙卡物理模型

## 摘要

本文记录 MAIGO 当前 CT 碳离子蒙卡中 TOPAS 参考模型与 GPU 模型的物理过程对应关系、数值实现和适用边界。TOPAS 采用 Geant4 modular physics list，对粒子逐步注册电磁、强子、离子核反应、衰变和停止过程；GPU 则采用 SYCL/CUDA 上的表格化带电粒子输运，使用 Geant4/TOPAS 预计算的 stopping power、核反应截面和相关末态 package，在 GPU 上采样并输运 C-12 及其带电碎片。GPU 模型覆盖了当前碳离子剂量的主要带电粒子链，但不是 Geant4 physics list 的逐过程复刻。中子、光子、电子的完整输运，以及普通衰变、放射性衰变和独立的 hadronic elastic 过程，在当前 CT production 配置中没有与 TOPAS 一一对应的 GPU 实现。

因此，本文中的“对应”分为三类：

1. **直接对应**：物理量和输运目的相同，但实现可能是 GPU 表格或近似模型；
2. **包驱动近似**：GPU 使用由 TOPAS/Geant4 生成的截面或相关末态 package，而不是运行时调用 Geant4 模型；
3. **当前未覆盖**：TOPAS 中存在，但当前 CT GPU 配置关闭或没有独立过程。

## 1. 参考模型与复现实验范围

### 1.1 TOPAS 参考配置

CT full-plan 使用 `Geant4_Modular` physics list。生成脚本
[`benchmark/ct/tools/fullplan/build_topas_full_plan.py`](../benchmark/ct/tools/fullplan/build_topas_full_plan.py)
写入如下模块集合：

```text
sv:Ph/Default/Modules = 7
  "g4em-standard_opt4"
  "g4h-phy_QGSP_BIC_HP"
  "g4decay"
  "g4ion-inclxx"
  "g4h-elastic_HP"
  "g4stopping"
  "g4radioactivedecay"
```

当前 CT 结果使用 TOPAS 4.2.p3 / Geant4 11.3.2。剂量由 `DoseToMedium` scorer 输出；LET 由 HadronLET 扩展的 `myHadronLET` scorer 输出。典型输入见
[`benchmark/ct/RT07575/conventional/fullplan_mc/run_full_plan.txt`](../benchmark/ct/RT07575/conventional/fullplan_mc/run_full_plan.txt)
和
[`benchmark/ct/RT06423/fullplan_mc/run_full_plan.txt`](../benchmark/ct/RT06423/fullplan_mc/run_full_plan.txt)。

这里的 `7` 表示七个 physics module constructor 都加入了所选 physics list。它不表示每个 primary C-12 history 都会实际触发七类过程：过程是否执行取决于粒子种类、能量、材料和粒子是否到达相应状态。例如 `g4radioactivedecay` 只对放射性核适用，`g4stopping` 主要处理停止/静止后的强子过程，而不是普通离子的连续 `dE/dx`。

### 1.2 GPU 参考配置

本文所称 GPU 参考模型是 README 里的 tightest production 数值，代表配置见
[`config/beam_ct_fullplan_rt07575_let_soft_tissue.yaml`](../config/beam_ct_fullplan_rt07575_let_soft_tissue.yaml)。其主要设置为：

```yaml
maximum_step_mm: 0.1
maximum_relative_energy_loss: 0.001
energy_cutoff_MeV: 0.1
secondary_local_deposit_cutoff_MeV: 0.1
enable_energy_straggling: true
enable_multiple_scattering: true
enable_primary_attenuation: true
enable_secondary_generation: true
enable_secondary_transport: true
enable_fragment_cascade: true
enable_neutral_transport: false
maximum_cascade_generations: 2
use_particle_specific_stopping_power: true
scorerLET: true
```

GPU 输运实现位于
[`src/transport_sycl_legacy.cpp`](../src/transport_sycl_legacy.cpp)，配置字段和物理开关定义于
[`include/carbon/transport_config.hpp`](../include/carbon/transport_config.hpp)。该配置中的
`dose_output_scale: 0.982` 是输出剂量校准因子，不属于底层物理过程；它不会改变原始沉积能量、核反应概率或输运轨迹。

## 2. TOPAS physics modules

| TOPAS module | Geant4 constructor | 主要物理作用 | 对 C-12 full-plan 的典型相关性 |
|---|---|---|---|
| `g4em-standard_opt4` | `G4EmStandardPhysics_option4` | 带电粒子电离和激发、电子/正电子过程、光子过程、离子电磁能损、多重散射等 | primary C-12 的连续能损和多重散射是剂量主项 |
| `g4h-phy_QGSP_BIC_HP` | `HadronPhysicsQGSP_BIC_HP` | 质子、中子、介子等强子非弹性过程，包含 BIC/QGSP/HP 相关模型组合 | 对 C-12 反应生成的强子和中性次级重要 |
| `g4decay` | `G4DecayPhysics` | 不稳定粒子的普通衰变 | 对短程带电碎片通常较小，但对相应不稳定粒子仍是完整 TOPAS 物理的一部分 |
| `g4ion-inclxx` | `G4IonINCLXXPhysics` | 离子-核非弹性反应，使用 INCLXX 离子反应模型 | primary C-12 和部分离子碎片反应的主要参考过程 |
| `g4h-elastic_HP` | `G4HadronElasticPhysicsHP` | 强子弹性散射，高精度中子/强子弹性模型 | 影响中性粒子和强子输运的角分布 |
| `g4stopping` | `G4StoppingPhysics` | 粒子停止后的强子俘获/静止过程 | 不是普通 C-12 连续 stopping power 的名称 |
| `g4radioactivedecay` | `G4RadioactiveDecayPhysics` | 放射性核的衰变链 | 对放射性碎片和长程活化贡献相关 |

Geant4 modular list 的模块名称与 constructor 对应关系见 TOPAS 官方的 [Modular Physics Lists 文档](https://topas.readthedocs.io/en/3.6.1/parameters/physics/modular.html)。

## 3. GPU 输运实现

### 3.1 带电粒子连续能损

GPU 对每个带电粒子采用步进输运。在一个 step 中，连续能损近似为

\[
\Delta E_{\mathrm{cont}} = S_{p,m}(E)\,\Delta s,
\]

其中 `S_{p,m}` 是粒子种类 `p`、局部材料 `m` 和能量 `E` 对应的 stopping power，`Δs` 是步长。该量由 CSV 表格插值，而不是在 GPU 上调用 Geant4 的 `G4ionIonisation` 或其他 Geant4 process。

在 `best` profile 中：

- C-12 主粒子使用 `stopping_power_file`；
- 碎片使用 `ion_stopping_power_*_geant4_11_3_2.csv` 中的粒子特异 stopping-power ratio；
- CT 中可按 air/lung/water/bone 使用材料 stopping-power 或质量缩放表；
- 不存在的碎片种类回退到水中相应表格或 C-12 effective-charge approximation，具体取决于开关和表格覆盖范围。

### 3.2 能损涨落

`enable_energy_straggling` 打开重粒子 condensed total-loss step-wise 涨落。方差使用碰撞运动学的 `Tmax/beta^2` 相对论项，并在低速极限退化到 Bohr 方差；采样限制为 `0..min(2*meanLoss,E)`。四个 100/200/300/400 MeV/u 单能水箱配置共享单位 scale，不再用非单调能量表补偿公式中缺失的相对论项。

可选的 `enable_step_stable_straggling`（默认 `false`）将 primary 涨落按固定物理块采样；`straggling_sampling_length_mm` 指定块长（例如 `0.1`）。启用后每个物理块复用一个由 `history_id + block_index` 确定的 Gaussian，并将块涨落按路径长度分配，因此完整均匀材料块的累计 mean/variance 不随 transport step subdivision 改变。材料/CT interface 仍由既有几何 clamp 保证不跨界；block 状态按物理路径连续，故 interface 会结束当前 transport step 但不会隐式重抽 block Gaussian。该模式当前只覆盖 primary；secondary 仍使用 legacy step-wise 语义。`false` 或块长 `0` 保留历史路径。

需要区分 primary 和 secondary：

- `enable_energy_straggling: true` 对 primary C-12 生效；
- 碎片能损涨落还需要单独打开 `enable_secondary_energy_straggling`；当前 CT production 配置没有打开该项。

因此，GPU 的能损涨落与 TOPAS `g4em-standard_opt4` 的完整随机过程具有相同物理目的，但不是同一套 Geant4 step process。当前实现把 continuous loss 与 unresolved hard electron transfer 凝聚在同一步内，没有显式 delta-electron transport，也没有完整复制 `G4IonFluctuations`/`G4UniversalFluctuation` 的 Gaussian/Gamma/Uniform/Glandz regime switch。

### 3.3 多重库仑散射

GPU 使用自定义的 Highland 投影 RMS 散射模型。其主要依赖粒子电荷、质量、`βp`、路径长度、局部密度和 radiation length：

\[
\theta_{\mathrm{rms}} \approx
\frac{13.6\,\mathrm{MeV}\,z}{\beta p c}
\sqrt{\frac{x}{X_0}}
\left[1+0.038\ln\left(\frac{xz^2}{X_0\beta^2}\right)\right].
\]

实现见 [`include/carbon/multiple_scattering.hpp`](../include/carbon/multiple_scattering.hpp)。

当前普通 CT production 配置设置 `enable_ct_material_mcs: false`，因此 CT 中默认使用历史 all-water radiation length；代码虽然支持 air/lung/water/bone 的材料 radiation length，但需要显式打开该开关。该差异是 GPU 与 TOPAS 电磁多重散射模型之间的重要近似边界。

### 3.4 C-12 核反应和 primary attenuation

primary 核反应概率使用宏观截面 `Σ(E,m)`：

\[
P_{\mathrm{int}} = 1-\exp[-\Sigma(E,m)\Delta s].
\]

`nuclear_cross_section_file` 及 CT 材料截面文件提供能量插值表。发生反应后，GPU 不在运行时调用 `G4IonINCLXXPhysics`，而是：

1. 按入射能量选择 reaction-package bin；
2. 随机选择一个预计算的相关末态；
3. 对末态粒子能量按实际反应能量进行比例调整；
4. 将带电碎片放入 secondary queue，将中子/光子送入可选 neutral queue 或按配置处理。

因此，GPU 的 primary reaction 是 **TOPAS/Geant4/INCLXX 结果驱动的 correlated-final-state surrogate**，而不是实时 INCLXX。

当前 CT production 默认使用：

- `reaction_package_file`：C-12 primary 相关末态；
- `cascade_package_file`：碎片级联相关末态；
- CT 材料专用 reaction/cascade package：只有 YAML 明确提供对应路径时才启用，否则回退到通用 package。

### 3.5 带电碎片输运和 cascade

反应产生的带电碎片通过 GPU secondary queue 批处理输运。每个碎片继续进行：

- 连续 stopping power 能损；
- 可选能损涨落；
- Highland MCS；
- 能量低于 `secondary_local_deposit_cutoff_MeV` 时的局部沉积；
- 由 cascade package 和 projectile-specific cross section 决定的后续核反应。

`maximum_cascade_generations: 2` 表示普通 CT production 最多继续两代碎片级联。级联 package 的末态保持相关性，未能继续输运的重反冲、未支持产物和队列溢出能量作为 residual local heat 计入沉积能量，以保持能量闭合。

### 3.6 中性粒子

GPU 代码支持中子/光子的 `first_interaction` 和 `full` 两种可选 package-based neutral transport，但普通 CT production 设置：

```yaml
enable_neutral_transport: false
```

并且 `neutral_local_kerma_fraction` 默认是 `0.0`。因此当前 CT best 不输运由核反应产生的中子和光子，也不自动将其全部能量计入剂量。只有显式启用 neutral package 和相关开关时，才会产生对应 GPU 中性粒子贡献。

### 3.7 电子、光子和原子退激发

TOPAS 的 `g4em-standard_opt4` 可以输运电子、正电子和光子，并处理 bremsstrahlung、湮灭、成对产生、Compton、光电效应等过程。当前 CT GPU 没有通用电子/光子 secondary queue，也没有完整的原子退激发输运。

GPU 可选的 `electronic_buildup_fraction` 和 LET delta-electron fraction table 是连续能损/LET 的局部近似，不等价于显式输运电子和 δ 射线。普通 CT production 的 `electronic_buildup_fraction` 默认关闭；LET scorer 使用的 delta-electron 表只改变 LET 定义相关的 restricted electronic stopping，不改变 dose transport。

### 3.8 衰变和静止过程

当前 GPU 没有 `g4decay` 或 `g4radioactivedecay` 的通用时间/寿命/衰变链模块，也没有独立的 `g4stopping` at-rest hadronic process。reaction package 中已经生成的带电末态会被当作当前支持的粒子输运，但不再自动模拟其后续放射性衰变。

## 4. 过程对应关系总表

| TOPAS/Geant4 过程 | GPU 对应项 | 当前 CT production 状态 | 等效性判断 |
|---|---|---|---|
| 电离、激发和离子连续能损 | stopping-power table + CT 材料/密度插值 | 开启 | 物理目的直接对应，数值实现为表格化 |
| 能损涨落 | Bohr straggling | primary 开启，secondary 默认关闭 | 近似对应 |
| 多重库仑散射 | Highland custom MCS | 开启；普通 CT 默认 all-water `X0` | 近似对应，材料 MCS 不是默认完全等效 |
| C-12 核非弹性 | C-12 cross section + reaction package | 开启 | package 驱动近似，不是运行时 INCLXX |
| 离子碎片生成 | reaction package secondaries | 开启 | 相关末态近似对应 |
| 碎片连续输运 | isotope-specific stopping power + MCS | 开启 | 直接物理目的对应 |
| 碎片核级联 | cascade package + cascade XS | 开启，最多 2 代 | package 驱动近似 |
| 强子弹性 HP | 无独立 hadronic-elastic process；部分角度效应由 MCS/末态表示 | 无一一对应 | 不完全覆盖 |
| 中子/光子强子与 EM 输运 | optional neutral package | CT best 关闭 | 当前不覆盖 |
| 电子/正电子/光子完整 EM 输运 | 无通用 EM secondary transport | CT best 关闭 | 当前不覆盖 |
| 普通衰变 | 无通用 decay queue | 关闭/未实现 | 当前不覆盖 |
| 放射性衰变 | 无 radioactive-decay chain | 关闭/未实现 | 当前不覆盖 |
| 停止/静止强子过程 | 无独立 at-rest hadronic process | 未一一实现 | 当前不覆盖 |
| `DoseToMedium` | voxel deposited-energy scorer，按局部质量换算 Gy | 开启 | scorer 层对应 |
| `myHadronLET` | dose-weighted LET moments，使用 restricted electronic stopping table | 开启 | 定义目标对应，输运实现不同 |

## 5. 剂量和 LET scoring

### 5.1 剂量

GPU 在 continuous step、secondary transport 和 reaction-local deposit 上累积
沉积能量，并按 voxel 质量换算为 dose-to-medium：

\[
D_v = \frac{E_{\mathrm{dep},v}}{m_v}.
\]

当前 reaction/cascade v1 package 的 local deposit 直接来自 TOPAS/Geant4
`G4Step::GetTotalEnergyDeposit()`。这避免了把“入射动能 - 产物动能”中的
反应 Q 值和核质量差误当作局部剂量。旧 package layout 不再兼容，loader
会直接报错。输出文件可以是 MeV、Gy CSV 或
MetaImage MHD/RAW；当前 CT 验证保持 `dose_output_scale=1.0`。

TOPAS 使用 `DoseToMedium` scorer。两者都以介质质量为分母，但由于 TOPAS 还显式输运电子、光子、中子和衰变产物，GPU 的 dose-to-medium 并不自动包含这些未覆盖贡献。

### 5.2 LET\(_d\)

TOPAS 使用 HadronLET 扩展的 `myHadronLET`，并通过 `WeightBy = "dose"` 输出剂量加权 LET。GPU 使用 `scorerLET: true` 后，按粒子、generation、材料和 voxel 累积 LET numerator/denominator moments，最终计算：

\[
LET_d = \frac{\sum_i L_i\,\Delta E_i}{\sum_i \Delta E_i},
\]

其中 `L_i` 使用 restricted electronic stopping power。GPU 对低于局部 transport cutoff 的带电粒子尾部继续补充 LET moment，避免高 LET 末端因局部沉积截断而系统性偏低。

该 scorer 的目标定义与 HadronLET 一致，但 GPU 不是逐步调用 TOPAS extension；它使用预计算 stopping-power 和 delta-electron correction table。对 all-hadron LET，中性粒子、电子/光子和未建模衰变产物的差异会直接影响比较结果。

## 6. `best`、`medium` 和 `fast` 的物理含义

本文的过程对应关系以 `best` 为准。其他 profile 是同一模型的精度/速度折中：

| Profile | 主要区别 | 适用场景 |
|---|---|---|
| `best` | 0.1 mm 最大步长、0.1 MeV secondary local cutoff、粒子特异 stopping power、LET | 最终 dose/LET 验证 |
| `medium` | 中等步长和次级近似，保留主要核反应和碎片级联 | 计划迭代和大规模敏感性分析 |
| `fast` | 0.5 mm 步长、1 mm secondary condensed step、2 MeV 次级局部沉积，通常使用 C-12 effective-charge approximation | 快速物理剂量预估；LET 仅适合趋势分析 |

这些 profile 不会改变 TOPAS reference physics list；它们只改变 GPU surrogate 的步长、截断、stopping-power 表和 scorer 选项。

## 7. 主要限制与误差来源

当前 GPU 与 TOPAS 的差异主要来自以下方面：

1. TOPAS 实时运行 Geant4 process，GPU 运行预计算 cross-section/reaction/cascade package；
2. 普通 CT best 默认使用 all-water MCS radiation length，而非每个 CT 材料的 Geant4 radiation length；
3. GPU 当前关闭中子/光子输运，且没有完整电子/光子 secondary transport；
4. secondary energy straggling、普通粒子衰变、放射性衰变和独立 hadronic elastic process 没有完全实现；
5. cascade generation 被限制为两代，超出支持范围的能量以 residual local heat 处理；
6. reaction/cascade package 的能量、深度、材料覆盖范围限制了泛化能力；
7. LET 对 fragment species、低能末端和中性粒子贡献比总 dose 更敏感。

因此，GPU 结果与 TOPAS 的高 gamma 通过率代表在当前病例、能量范围和 scoring definition 下的验证结果，并不表示 GPU 已经成为完整 Geant4 physics list 的替代品。跨病例使用时应继续验证：入口剂量、Bragg peak/R80、fragment tail、异质材料界面、all-hadron LET 以及不同随机种子之间的统计稳定性。

## 8. 可复现性检查

### 8.1 检查 TOPAS 模块是否加入

确认输入中存在：

```text
sv:Ph/Default/Modules = 7 "g4em-standard_opt4" "g4h-phy_QGSP_BIC_HP" "g4decay" "g4ion-inclxx" "g4h-elastic_HP" "g4stopping" "g4radioactivedecay"
```

若要打印每种粒子实际注册的 process，可在 smoke 输入中加入：

```text
b:Ph/Default/ListProcesses = "True"
```

`Modules = 7` 只能证明七个 module constructor 被加入，不能代替 per-particle process dump。

### 8.2 检查 GPU 开关

运行前确认配置中至少包含：

```yaml
enable_primary_attenuation: true
enable_secondary_generation: true
enable_secondary_transport: true
enable_fragment_cascade: true
enable_neutral_transport: false   # 当前 CT best 的明确选择
use_particle_specific_stopping_power: true
scorerLET: true
```

GPU 启动输出中的 backend 标签应包含 `physics-best`、`multiple-scattering`、`attenuation`、`secondary-generation`、`secondary-transport`、`fragment-cascade`，并在 LET 运行中包含 `letd-scoring`。

## 9. 结论

TOPAS CT full-plan 使用的七个 Geant4 modules 全部加入了参考 physics list，但每个 module 只对适用粒子和状态实际生效。GPU 当前准确覆盖的是对碳离子剂量最重要的带电粒子链：电磁能损、MCS、primary C-12 非弹性、带电碎片输运和有限代数 cascade；其核反应和碎片末态由 TOPAS/Geant4 预计算 package 驱动。中性粒子、电子/光子 secondary、衰变和 at-rest 过程仍是与 TOPAS 的主要非等效部分。

该模型适合在明确的能量范围、CT 材料映射、剂量/LET scorer 定义和 package 覆盖范围内进行高吞吐 GPU 验证。若目标是把 GPU 变成完整 Geant4 physics list 的替代实现，下一步应优先补充材料条件化的 MCS、secondary straggling、中性粒子输运和衰变，而不是只增加 primary histories。

## 10. 可选物理开关与速度门控

前三项扩展已经是运行时开关，关闭时不分配中性队列、不加载中性 package，也不执行材料 radiation-length 查询或 secondary Bohr 随机采样：

```yaml
enable_ct_material_mcs: false
enable_secondary_energy_straggling: false
enable_neutral_transport: false
neutral_transport_mode: first_interaction  # 仅在 neutral=true 时生效
```

对应的独立 smoke 配置见
`config/beam_ct_physics_extensions_smoke.yaml`。该配置默认把三项都打开，生产 `best/fast` 配置则显式写出 secondary straggling 为 `false`，避免把默认值误认为物理过程已经启用。

1,000 histories 的 smoke 主要测启动与分配开销，不能用于估计稳态吞吐。使用同一 37° TPS-source CT smoke、NVIDIA TITAN RTX、10M histories，并将队列扩大到 `secondary=48M`、`neutral=24M` 以保证 `overflow=0`，得到：

| 开关组合 | 运行时间 | 吞吐 |
|---|---:|---:|
| 三项关闭 | 109.708 s | 91.15 k histories/s |
| 仅材料条件化 MCS | 110.509 s | 90.49 k histories/s |
| 仅 secondary straggling | 119.343 s | 83.79 k histories/s |
| 仅 neutral first-interaction | 113.657 s | 87.98 k histories/s |
| 三项开启 | 121.832 s | 82.08 k histories/s |

相对于三项关闭，10M 稳态吞吐损失分别约为材料 MCS 0.7%、secondary straggling 8.1%、neutral first-interaction 3.5%，三项同时开启 10.0%。每次运行的 backend 标签会记录 `ct-material-mcs`、`secondary-straggling` 和 `neutral-transport-first-interaction`，可以用于结果审计。高统计运行不能继续使用 100k 的队列容量，否则会截断二级粒子并使剂量不适合验证。

在三项全开启、无队列溢出的条件下，两个独立 1M seed 的 37° CT 体素剂量总和差异为 0.010%；剂量高于峰值 10% 的体素中，绝对相对差异的中位数/95 百分位为 0.34%/1.11%。同一 seed 的 1M 与 10M（按 histories 归一化）总和差异为 0.51%，说明 1M 已可用于稳态速度和基本统计检查，但 10M 更适合作为最终 benchmark。

衰变目前不设置一个会被静默忽略的 YAML 开关。现有 cascade package 没有携带衰变寿命、分支和 daughter kinematics，GPU 也没有 decay queue；在取得同版本 TOPAS 的 decay package 后，应以独立 `enable_decay` 开关接入，并在开启时强制要求 package 文件和生命周期表存在。

## 参考资料

1. TOPAS, *Modular Physics Lists*, https://topas.readthedocs.io/en/3.6.1/parameters/physics/modular.html
2. Geant4 Collaboration, *Physics Reference Manual*, https://geant4.web.cern.ch/documentation/
3. Villadslj, *Topas-Extension*（HadronLET scorer）, https://github.com/Villadslj/Topas-Extension
4. MAIGO GPU transport configuration: [`include/carbon/transport_config.hpp`](../include/carbon/transport_config.hpp)
5. MAIGO SYCL transport implementation: [`src/transport_sycl_legacy.cpp`](../src/transport_sycl_legacy.cpp)
