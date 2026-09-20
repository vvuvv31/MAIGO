# 材料与方法

修订日期：2026-09-20。描述当前工作树及指定配置，不视为冻结发布、临床准入或通用 Geant4 等价声明。[English](mm.md)。
历史数值只绑定对应的 executable、数据和配置，不能作为所有当前开关的验证。

## 1. 计算框架与当前配置

MAIGO 是 C++20 / SYCL 带电离子凝聚历史 Monte Carlo，在本地 RTX 2080 Ti
（`sm_75`）运行。GPU 查询 TOPAS/Geant4 派生表并重放核事件，不在线执行核内级联。
患者实现为 [transport_sycl.cpp](src/transport_sycl.cpp)，串行后端只支持子集。

参考采用 TOPAS 4.2.p3 / Geant4 11.3.2，含 standard_opt4 EM、QGSP_BIC_HP、
ion-INCLXX、elastic、stopping 和衰变模块。模块名不等于每个 projectile 实际挂载的模型；
已观察到 C12 使用 INCLXX。全离子弹性比较还需第 5 节所述的匹配扩展。
提取源码、编译器与冻结哈希见 [extensions/](extensions/README.md)。

当前有两类配置：

1. **宽束 / CT 生产。** 统一 EM 生产入口为[水](config/unified_water_production.yaml)和
   [RT07575](config/rt07575_unified_em_production.yaml)。
2. **Copper minibeam 验证。** 如
   [250 MeV/u、256k](config/beam_minibeam_field3cm_copper_e250_256k.yaml)
   （及 150/300 MeV/u 同类）启用 Copper 准直器与水中输运。这是研究验证预设，不是临床 TPS。

| 设置 | 水 / RT07575 生产 | Copper minibeam 场配置 |
|---|---|---|
| `em_model` | `g4_material_joint_v1` | `g4_material_joint_v1` |
| `enable_secondary_unified_em` | `true` | 未写，默认 **false** |
| 原发 / 次级涨落 | 均开启，`straggling_scale: 1.0` | 原发开启；次级涨落未写，默认 false |
| `multiple_scattering_model` | 显式 `highland`（代码默认） | `fermi_eyges`，`fermi_eyges_species: all_charged` |
| `ct_secondary_exact_faces` | `true`（CT） | 非 CT 网格 |
| `secondary_species_grouping` | `true` | 不是 minibeam 性能目标 |
| `secondary_step_chunking` | `true`：16 次完整循环后压紧存活队列 | 按配置 |
| EM 查表 | `CARBON_EM_EXACT_INDEX=ON` | 同一构建 |
| 非弹性 / 次级输运 | 开启；代数上限 2 | 水中代数上限 2；Copper cascade 代数 3 |
| 独立全离子核弹性 | 生产 YAML 不载入 | Copper 离散弹性关闭 |
| 电子剂量 | 抽样 δ 能量局部沉积，不运行电子能量包 tracking | 相同 |
| Minibeam 专属能损缩放 | 不适用 | 仅原发 C12：`minibeam_water_primary_stopping_power_scale: 0.9958` |

其中 `primary_em_model: legacy` **不代表此时仍使用旧 stopping**：
`em_model: g4_material_joint_v1` 选择统一路径，原 primary-only 选项不是叠加模型。
反过来，只写 `em_model` 也不会自动开启次级统一 EM；配置结构中的次级开关默认 false。
更新 executable 不会自动升级所有历史 / 研究 YAML。
GPU 上 Legacy EM（`em_model: legacy`）已被拒绝；serial/cpu 后端（含 `transport_cpu`
与 CPU 测试）保留 legacy 路由。Joint water EM（`g4_joint_water_v1`）实现已删除，
其 YAML 键会被拒绝。

统一 EM 和种类分组已获准接入 CT/水生产路径；低密度 production-cut 区、患者 Gamma，
以及分组引起的剂量差异仍待调查。执行质量接受与物理精度验收是不同结论。
Minibeam 场匹配是另一项未完成的验证（第 11 节）。

GPU 性能与 minibeam 开发使用 FP32 剂量记分（`CARBON_DOSE_FP32=ON`）。
FP32 原子累加顺序差异是预期现象，单独不能作为否决候选的理由。

## 2. 材料、源与几何

CT HU 映射到实际体素密度及 25 个 Schneider 组分区之一，使用 13 种元素组成。
CT 不走四分类回退。水采用独立、固定 SHA 的 G4_WATER，不冒充某个 Schneider 分区。
统一 EM 包覆盖水、25 分区及 18 种带电离子；密度节点查询保留材料 stopping 和阈值特性。

源由 TPS spot CSV histories、能散、发射度、扫描磁铁和患者摆位定义。分片前应与参考匹配
每个 spot 的整数 histories。TOPAS passive RotZ 使用 `R(+RotZ) × (world − Trans)`，
方向只旋转不平移。CT 打包、源摆位、剂量映射分别核对，不采用剂量拟合配准。
见[源与几何](docs/planning.md)。

Minibeam 场几何为 15 缝 Copper 准直器（缝宽 0.5 mm、节距 3.6 mm、厚度 60 mm），
下游为水模体。GPU 经空气和 Copper 再进入水；absorbing-geometry 回放可从水面内侧开始。
Copper 密度 8.96 g/cm³、质量辐射长度 12.8628 g/cm² 为显式参数。Copper stopping、
非弹性率和分能量碎片 cascade 包是 TOPAS/Geant4 11.3.2 提取表，不是 Schneider CT 分区。

## 3. 电磁过程

### 3.1. 当前统一模型的一步

2026-09-15 用户验收后，水和 Schneider 全部 18 种已支持离子采用两矩 Gamma δ 聚合与解析分步修正。

1. 按物种、材料、密度和动能准备受限 stopping/range、离子修正、原生涨落与 δ 矩。
2. 按 3.1.1 节选择当前步长。
3. 计算受限平均能损，在线性分支加入下述 Poisson 分步修正；range 反演分支不重复修正。保留原生受限涨落，scale=1。
4. 按本步 δ 总损失均值与方差抽一次 Gamma，扣除能量并局部沉积；不抽逐电子碰撞，也不使用 δ 时钟限制步长。
5. 推进粒子、应用配置的 MCS，处理核候选并排入产物。核光学深度机制保持独立。

随机流仍绑定粒子身份，保留 Philox 与跨续跑计数；上述 1% 是平均损失估计限制，不是随机损失硬上限。

仅 minibeam 原发 C12 水路径在抽样总损失后乘冻结因子 0.9958，并裁剪到可用动能。
同一次 draw 的 `continuous` 与 `delta` 按 `deposited/unscaled` 同比例缩放，以保持分区。
该因子不作用于次级 C12、碎片、Copper 或宽束生产预设。相对当前 TOPAS 参考，
250 MeV/u 全链 A/B 在 2-D/IDD L1 和 Bragg 深度上支持 0.9958 优于 1.0；
这是相对该参考的表/抽样修正，不是改写 Geant4 stopping 表。

#### 3.1.1. 当前步长如何决定

GPU 先提出电磁步长，再按几何和可能的核碰撞裁短。能损与 MCS 使用最终的 `h`。
实现见 [unified_em_view.hpp](include/carbon/unified_em_view.hpp) 的 `step_from_range`
以及 [transport_sycl.cpp](src/transport_sycl.cpp) 的原发/次级循环。

**统一 EM（水、RT07575 和当前 minibeam 场 YAML）。** YAML 的 `maximum_step_mm` 和
`maximum_relative_energy_loss` 被 Geant4 原生 StepFunction **替换**，不是再取 min。
剩余受限 CSDA range \(R\) 来自包内样条，用步首能量和局部密度求值。参数为
`f = dRoverRange`、`r_final = finalRange`，按离子存为 `step_fraction` 和 `final_range`：

```text
h_EM = f R + r_final (1 − f) (2 − r_final/R)，R > r_final
h_EM = R，                                 其他情况
```

v1 包中全部离子 `f = 0.1`。`r_final` 随物种变化，提取自
`G4VEnergyLossProcess::finalRange`：

| 离子 | `r_final` |
|---|---|
| p | 0.05 mm |
| d、t、He-3、He-4 | 0.02 mm |
| He-6 及 \(Z\ge 3\)（含 C12） | 0.001 mm |

远离停止时 \(h_{\mathrm{EM}}\approx fR\)（C12 约为剩余 range 的 10%，外加约 0.002 mm）。
在 \(R=r_{\mathrm{final}}\) 两支的值和一阶导数连续（\(h=R\)，\(h'=1\)），步长不会跳变。
\(R\le r_{\mathrm{final}}\) 时提议步等于剩余 range。这是电磁几何提议，不是固定毫米步，
也不是 YAML 相对能损上限。高能 C12 步可以到许多毫米，近停止才变成微米量级。

δ stopping 大于零时再要求

```text
h ≤ 0.01 T / (S0 + D0)
```

限制的是**平均**受限加 δ 总损失，不是抽到的随机损失。这是步长上限，不是后面选择连续均值分支的 `linear_limit`（3.2.1 节）。

**仅 Legacy EM。** 提议步为
`min(maximum_step_mm, maximum_relative_energy_loss × T / S)`。
GPU 生产拒绝 `em_model: legacy`；serial/CPU 仍保留该路径。

**随后将 `h` 取下列最小值：**

- 一维记分深度面（水和 minibeam 都有；当前 minibeam YAML 为 0.25 mm 分箱）；
- 轨迹在 CT 网格内时的精确 CT 体素面；
- 启用时的 slab、insert、模体和 Copper 材料边界；
- 本候选步内的核碰撞：剩余光学深度 \(\tau=-\ln U\) 在 `h` 上消耗；若 \(\Sigma h \ge \tau\)，
  则把 `h` 收到 \(\tau/\Sigma\) 并标记碰撞，否则只减 \(\tau\) 继续走。
  碰撞距离不是事先单独算出的唯一上限。

横向记分体素在生产中**不**卡输运（`voxel_scorer_clamps_transport` 关闭）。
Fermi–Eyges 内部 0.1 mm 分段是在已经选定的 `h` 里再切 MCS，不替代 StepFunction。

**输运停止**于动能落到 `energy_cutoff_MeV`、飞出模体，或达到原发/次级最大步数。
cutoff 处残余能量记在当前体素。

### 3.2. 受限平均能损与密度

令 `T` 为粒子**总动能**（MeV，不是 MeV/u），`E=T/A` 为每核子能量，`h` 为 3.1.1 节已经选定的步长（mm）。
连续能损是这一步上的受限电离，不是另一套步长公式。

#### 3.2.1. 原发 / 次级连续能损如何形成

`h` 确定之后，电离按材料电子生产阈切开：

- 阈**以下**的能量转移 → 受限 / **连续**（`draw.continuous`）；
- 阈**以上**的能量转移 → 聚合 **δ**（`draw.delta`，3.3 节）。

两块分开抽样再相加。连续损失不是 \(S_{\mathrm{total}} h\)，也不是 3.1.1 节 YAML 的 `maximum_relative_energy_loss`。

**统一 EM**（`em_model: g4_material_joint_v1`）对原发离子、以及已打开统一 EM 的次级，用同一套「先受限均值、再涨落」：
包内存原生受限 stopping、range 和 inverse-range 样条。相邻密度节点先按（实际密度 / 节点密度）求值再插值，
**不能**写成水 stopping 乘 \(\rho/(1\,\mathrm{g/cm^3})\)。包还给出 Geant4 原生 `linLossLimit`，按离子存为 `linear_limit`。
从 `unified_em_v1.bin` 读出（3,150 条材料/离子记录；同一离子的 175 个密度节点共用一个值）：

| 离子 | `linear_limit` |
|---|---|
| p、d、t（\(Z=1\)） | 0.01 |
| He-3 及更重（含 C12） | 0.02 |

这**不是**毫米步长。「小步 / 大步」指估计受限损失 \(S_0 h\) 占当前总动能 \(T\) 的比例大不大，
也**不是** 3.1.1 节的 1% 步长上限 \(h\le 0.01\,T/(S_0+D_0)\)（那道用受限 **加** δ stopping，先限制 \(h\)）。
C12 走线性支的条件是 \(S_0 h \le 0.02\,T\)。`primary_restricted_mean_candidate` 里的缺省 `0.02` 在统一路径上不会单独生效，调用传入的是 `r.linear_limit`。

1. 按物种、材料、局部密度和 \(T\) 查出受限 stopping \(S_0\)、剩余受限 range \(R\)、inverse range 和离子修正量。
2. **小步（线性支）：** \(h<R\) 且 \(S_0 h \le\) `linear_limit` \(\times T\)。受限均值从 \(S_0 h\) 出发，再加 3.3 节 Poisson 分步修正。
   **大步（range 反演）：** 估计受限损失超过该比例，或 \(h\ge R\)。末端能量由 \(R-h\) 反演得到（\(h\ge R\) 则本步停掉，连续均值就是剩余 \(T\)）。
   该支不再做分步修正。反演保持步首质量 / 电荷缩放。
3. 离子修正在中间能量查询。该中点包含 δ 平均损失，
   \(T_{\mathrm{mid}}=\max(0.5T,\,T_{\mathrm{mid}}-\tfrac12 D_0 h)\)，\(Z>2\) 还有低能替换。
   这不是旧的**总** stopping 预测中点。
4. 均值裁到可用动能；落到包最低动能可停止径迹。
5. 若开启能损涨落，在该受限均值上抽 IonFluc 或 Universal/Urban（当前生产预设 `straggling_scale: 1.0`），得到 `draw.continuous`。
   未开涨落则连续损失就是该均值。

仅 minibeam **原发 C12 水路径**在抽完 `continuous+delta` 后乘冻结因子 0.9958，两块再按 `deposited/unscaled` 同比例缩放。
次级 C12、碎片、Copper 和宽束生产预设不用这个因子。

**哪些径迹走上面这一套。**

| 径迹 | 连续能损 |
|---|---|
| 水或 Schneider CT 中的原发 C12，统一 EM | 上述流程 |
| `enable_secondary_unified_em: true` 的次级（水和 RT07575 生产 YAML） | 同一套 |
| 当前 minibeam **场** YAML 的水中次级 | 该键未写，默认 **false**。粒子特异水表中点；均匀水表已是绝对 stopping，**不再乘密度**。中点每核子能量为 \(\max(0.01\,\mathrm{MeV/u},\,(T-\tfrac12 S h)/A)\)。若开次级涨落，packaged 涨落只作用于次级 C12。`minibeam_water_secondary_c12_enable_unified_em` 可把**仅 C12**切到统一 EM，默认关。 |
| Copper 准直器（原发 C12 与碎片） | 不在统一包内。提取的 Copper 表，先预测再中点 \(S\)，可选凝聚涨落。 |
| 未开统一 EM 的 CT 次级 | 水表，步首与中点用同一套质量阻止本领因子或密度缩放。 |
| 未注册重反冲 | 专用 recoil stopping；generic recoil 另有 5% 相对能损卡步。 |

实现见 [unified_em_view.hpp](include/carbon/unified_em_view.hpp) 的 `mean` / `unified_em_loss`，
以及 [transport_sycl.cpp](src/transport_sycl.cpp) 的原发 / 次级循环。

旧 `ct_primary_midpoint_stopping` 与材料特异次级 stopping 仍供非统一路径使用，表中保存：

```text
S1(section,ion,E) = S_extracted / [rho_reference/(1 g/cm³)]
S(T,section,rho) = S1(section,ion,T/A) × rho/(1 g/cm³)
```

这里密度没有重复计算。但这些旧表 / 开关不决定当前统一模型的受限平均能损。
未注册的额外重反冲仍用专门的 stopping 路径。见[统一查询与能损](include/carbon/unified_em_view.hpp)。

He-4 等总动能超过 Geant4 默认 600 MeV EM 表上限的离子，必须显式提高 `EMRangeMax`
（2026-09-19 水片战役使用 10 GeV）。600 MeV 表不能作为 He-4 300 MeV/u 参考。

### 3.3. 受限涨落、δ 聚合与分步修正

`S0` 为受限 stopping，`D0=M1` 为 δ 平均能损率；`lambda_native` 为密度缩放后的原生 GetLambda。
受限连续涨落继续使用适用的 IonFluc 或 Universal/Urban 采样器。对受限线性分支：

```text
x = lambda_native h
F(x) = 1 - 2/x + 2(1-exp(-x))/x²    [F(0)=0]
S_eff = S0 - h/2 (S0 F(x) + D0) dS0/dT
mu_delta = h max(0, D0 - (mean_restricted + D0 h)/2 dD0/dT)
v_delta = h M2(T)
Gamma shape = mu_delta²/v_delta; scale = v_delta/mu_delta
```

`F` 仅作用于连续自漂移，δ 跳变造成的漂移保持完整；离子修正中间能量也包含 δ 平均损失。
这是常率、局部线性的 Poisson 分步近似，用来补偿取消离散 δ 分步后的平均损失偏差，并非拟合 TOPAS 的系数。
均值/样本均受可用动能约束，零矩分支不抽样；Gamma 不保留离散零碰撞概率和更高阶矩。
矩表由原生能谱及 spin、form-factor、magnetic veto 的接受概率积分生成；方差采用步首 M2。
受限 stopping 排除的阈值以上损失只在 δ 聚合中计入一次。阈值依赖材料/密度，不能统一套用水阈值。
没有电子空间 tracking，也不叠加旧电子响应。`em_macro_ticks` 不是当前支持的开关。

统一 minibeam 水路径中 `use_water_electron` 关闭（`EmMode==1`）。
因此抽样的 `draw.delta` 留在输运离子的局部沉积中。这是与 TOPAS 显式电子的
production-cut 映射，不是载体标签逐项相等。诊断 ROI 从同一次 draw 记录
`continuous_sampled`、`delta_sampled`、`continuous_after_scale`、
`delta_after_scale` 和 `local_total_deposit`，不改变 RNG 与输运终态。

### 3.4. 库仑多重散射

两种模型都在**本步 Unified EM 能损抽完之后**才改方向/位置，核光学深度独立。
它们共用同一套质量辐射长度 \(X_0\)：均匀水用 G4_WATER，CT 用 Schneider 25 分区表，
再乘局部密度。实现见
[multiple_scattering.hpp](include/carbon/multiple_scattering.hpp)、
[sycl_device_math.inc](src/detail/sycl_device_math.inc)，
调用在 [transport_sycl.cpp](src/transport_sycl.cpp)。

两种实现共用一个 YAML 选择器（[宽束 FE](docs/broad_beam_fermi_eyges.md)）：

```yaml
enable_multiple_scattering: true
multiple_scattering_model: highland   # 或 fermi_eyges / fe
fermi_eyges_species: c12              # c12 / c12_he4 / c12_he4_pdt / all_charged
fermi_eyges_parameter_set: species_water
fermi_eyges_max_segment_mm: 0.1
```

代码默认 `highland`。水和 RT07575 生产 YAML 显式写 Highland，避免旧运行被静默切换。
Minibeam 场 YAML 当前选择 `fermi_eyges` 且 `all_charged`。只要出现
`multiple_scattering_model`，它就是权威开关，并关闭旧的 minibeam-only FE 键。

#### Highland：步末纯角度踢

投影角 RMS：

```text
t = rho × (h/10) / X0_mass
C = max(0, 1 + 0.038 ln(t Z²/beta²))
theta0 = 13.6 MeV × Z/(beta p c) × sqrt(t) × C
```

再可乘 `multiple_scattering_scale`（生产默认 1.0）。Minibeam 水中原发 C12 还有低能过渡：
\(E<180\,\mathrm{MeV/u}\) 时把 \(\theta_0\) 线性压到 0.20 倍。旧 minibeam Highland
还可把一部分方差放到更宽的高斯 core/tail 混合里，那不是 FE。

两个独立 Box–Muller 样本给出 \(\theta_x,\theta_y\sim\mathcal{N}(0,\theta_0^2)\)，
转到粒子坐标系后**只改方向**。本步空间位移仍沿**散射前方向**直线走：

```text
position += direction_old × step
```

没有横向位移，也没有本步的 \(y\)–\(\theta\) 相关。几何上这一步是直线段，新方向从下一步才生效。
这不是 Geant4 Urban/Wentzel 完整 msc，也不能替代强相互作用核弹性。

#### Fermi–Eyges：相关位移 + Poisson 尾

`ion_fermi_eyges_transport_step` 把物理步切成不超过
`fermi_eyges_max_segment_mm`（0.1 mm）的内部段。段中点能量按本步已抽的 \(dE\) 线性插值：

```text
E(s) = E0 − (s + h/2)/L × dE
```

每段调用 `water_ion_fermi_eyges_tail_step`，下一段用新方向。返回的位移相对
「沿入射方向直线漂移」。调用方随后：

```text
position += direction_old × L + displacement
direction = new_direction
```

`fermi_eyges_species` 之外的离子仍用 Highland。C12 使用冻结水候选
`(core, rate, tail) = (9.9 MeV, 0.0025 mm⁻¹, 2.4 MeV)`。p/d/t/He-4 使用
50/150/300 MeV/u 独立纯水 TOPAS 拟合，线性插值
（[标定](benchmark/fermi_eyges_species_water/calibration_manifest.json)）；
区间外保持端点值。He-3 和更重碎片仍回退到 C12 常数。50–300 MeV/u 区间不覆盖
主导部分 Bragg 碎片剂量的 `<50 MeV/u`。超出 `c12` 的范围仍属实验性，尤其在 Schneider 材料上。

**高斯 core。** 散射功率用当前材料 \(X_0\)：

```text
T = (Es × Z/(beta p))² / X0_mm
Var(theta) = T L
y = (L/2) theta + eta,   Var(eta) = T L³ / 12
```

因此 \(\mathrm{Cov}(y,\theta)=T L^2/2\)。\(\theta_x,\theta_y\) 与独立的
\(\eta_x,\eta_y\) 均为 Box–Muller 高斯。`Es` 即 `core_MeV`。

**Poisson 尾。** 事件数 \(N\sim\mathrm{Poisson}(\lambda L)\)。\(\lambda\) 按水中每毫米拟合，
再按水/\(X_0\) 比缩放到当前材料。Knuth 抽样不截断事件数。每个事件在段内均匀，
踢角 \(\mathcal{N}(0,(E_{\mathrm{tail}} Z/(\beta p))^2)\)，再用剩余路程把角变成额外位移。
0.1 mm 水步上均值约 \(2.5\times10^{-4}\)，多数步 \(N=0\)。

**诊断平面。** 若记分面落在本步内，在**已经抽好的段末态**上对 \((\theta(t),y(t))\)
做 Brownian bridge，不用起终点线性插值；线性插值会把方差弄成 \(f^2 TL\) 而不是 \(fTL\)
（[failed.md](failed.md)）。

| | Highland | Fermi–Eyges |
|---|---|---|
| 本步横向位移 | 无 | 有，且与末态角相关 |
| 角分布 | 单高斯 \(\theta_0\) | 窄 core + 稀有宽尾 |
| 步内能量 | 整步用步首 \(E\) | 0.1 mm 段、段中点 \(E\) |
| 位置更新 | \(\mathbf{r}+\hat n_{\mathrm{old}} L\) | \(\mathbf{r}+\hat n_{\mathrm{old}} L+\boldsymbol{\delta}\) |
| 不是什么 | Geant4 完整 msc | Urban 肩部；1 mm 薄片分位数未完全拟合 |

Minibeam Copper 走单独的 `fermi_eyges_tail` 入口（scale 1.0；碎片 MCS scale 0.785），
过程同类。已失败的 MCS/电子展宽路线列于 [failed.md](failed.md)，不得恢复。

见[MCS](include/carbon/multiple_scattering.hpp)。

## 4. 非弹性核过程

**MAIGO 的处理：先抽取碰撞，再重放 TOPAS 派生的相关末态事件。**

1. **确定碰撞位置。** 局部宏观反应率为 `Sigma = rho × sum(元素质量反应率 partials)`。
   原发 C12 沿路径消耗抽样光学深度 `tau = -ln(U)`，由剩余光学深度确定碰撞距离；
   已支持次级使用各自 projectile 的反应率。平均自由程是统计尺度，不是固定碰撞距离。
2. **选择靶元素与事件。** 按元素 partial-rate 比例选靶，在 EM 能损后的能量处选择相邻
   能量节点，再抽取一个完整 CINEL03 事件。保留产物能量与角度的关联，
   用同一个随机方位角将整个事件旋转到入射坐标系。
3. **输运末态产物。** 用抽样末态替代入射轨迹，已支持的带电碎片进入 GPU 队列，
   继续电磁输运及符合条件的后续非弹性反应。不支持通道、能量截断和代数限制显式记账；
   队列 overflow 使运行无效。

当前最低要求是固定的 Schneider v2.1 数据栈，次级核 registry 覆盖 14 种 projectile。
缺通道不以近邻靶替代，不整体缩放产物动能。原发和次级的 post-EM null candidate
保留轨迹和剩余动能，不重放核事件、不将剩余能量作为核反应局部沉积。
原发分支清除碰撞标志，并完成当前 EM 步及其剂量记分。
原先的解析碎裂备用路径及经验参数已移除；核输运必须使用经过验证的
Schneider/统一水 CINEL03 路径。详见[清理报告](docs/fred_cleanup.md)。
水中冻结 generation 设置为 2；He6/B8/C10 遵循声明的 EM-only 核策略。

Copper minibeam 核反应使用提取的 C12+Cu 率及分能量碎片 cascade 包（填补后的 INCL++ 样本）。
Copper cascade 代数为 3；水中 generation 命名空间独立，Copper 存活者进入水时从 generation 0 开始。
Copper 离散 General Ion Elastic 关闭。

**与 TOPAS 的区别**

| 比较项 | MAIGO | TOPAS / Geant4 参考 |
|---|---|---|
| 碰撞概率 | 插值提取的元素反应率表，并限定有效域 | 使用配置物理列表的截面数据与过程步进 |
| 核末态 | 从离散能量节点的有限相关事件库抽样 | 在相互作用状态调用适用核模型，生成产物 |
| 模型执行 | 不在线计算核内级联 | 参考 C12 已观察到调用 INCLXX；实际模型随 projectile 和能量变化 |
| 后续输运 | 已支持带电物种及有限核反应代数；中性/衰变范围受限 | 按启用的粒子过程和跟踪截断继续输运产物 |

复用 TOPAS 派生事件保留了抽样事件内部关联，并省去在线核模型计算。
这不意味着两套引擎等价：事件库统计量、能量节点抽样、步进和次级覆盖仍有差异。
事件共用方位角旋转加入于 9 月 5 日冻结 benchmark 之后。
见 [CINEL03 查询](include/carbon/inelastic_package_v3.hpp) 与 [GPU 输运](src/transport_sycl.cpp)。

## 5. 弹性核过程

**研究配置已支持原发 C12 在内的全部 18 种带电粒子、13 个 Schneider 靶元素，覆盖 CT 和统一水。** 默认生产配置仍保持无独立核弹性，等待匹配参考验收。库仑 MCS 是独立的电磁过程。

1. **确定碰撞位置。** 原发按弹性与非弹性宏观率之和消耗抽样核光学深度，再按反应率选择类型。次级弹性距离与非弹性距离竞争，弹性不受非弹性代数上限限制；几何、连续能损和 cutoff 可进一步缩短步长。
2. **抽样靶与动量转移。** 按材料元素 partial-rate 选靶，按率加权抽样相邻能量节点，读取 TOPAS 联合样本中的靶同位素、核质量和 `t/tmax`。使用当前能量与随机方位角，通过相对论两体运动学生成守恒末态，不对全部靶假设各向同性。
3. **输运反冲。** 既有 18 种物种中的反冲进入通常的带电队列；额外天然靶反冲使用覆盖 37 同位素的材料特异总 stopping 与 MCS，不继续显式核反应。总 stopping 包含凝聚核 stopping；cutoff 以下残余能量局部沉积。队列溢出或必需数据缺失使运行失败。

| 比较项 | GPU 研究实现 | 匹配 TOPAS 参考 |
|---|---|---|
| 反应率与末态 | 有限材料率表、靶同位素/动量转移样本 | 在线截面查询与 Geant4 模型求解 |
| 弹性模型 | p: hElasticCHIPS；d/t/He3/alpha: hElasticLHEP；其他既有离子: NNDiffuseElastic | 启用 CarbonIonElasticPhysics，挂载相同弹性模型 |
| 反冲覆盖 | 既有18物种及额外 EM-only 靶反冲 | 对生成粒子继续适用的物理过程 |
| 库仑散射 | 凝聚 Highland 或 FE | 配置的 Geant4 电磁过程 |

**旧 topas10x 不能直接作为新模型的匹配参考。** 审计确认旧列表只有 p、d、t、He3、alpha 有核弹性，GenericIon/C12 未挂载；新增 `CarbonIonElasticPhysics` 后补齐。匹配重跑保留原电磁、非弹性、stopping、衰变模块以及 CT、源、3D 剂量网格和总 histories，仅省略 LET scorer。

基准库为137能量节点、每节点512末态。运动学、host/device查询、水闭合及RT07575的6481909 histories分片验证通过且零溢出；2048样本及加密网格候选也通过分片闭合。低能截面起始区插值、额外反冲贡献及全统计匹配剂量验收仍未完成，尚未切换默认生产配置。

使用 [CT研究配置](config/rt07575_elastic_research.yaml) 或 [水研究配置](config/unified_water_elastic_research.yaml)，不是旧 `enable_nuclear_elastic` 开关。2026-09-11 已接受包所用的冻结 AllIonElastic 源码在
[extensions/topas/all_ion_elastic/frozen_production](extensions/topas/all_ion_elastic/frozen_production)。
详见[实现与数据](docs/all_ion_elastic.md)。

## 6. 电子与中性产物

当前统一 EM 抽样聚合 δ 损失，但不空间跟踪这些电子。受限能损与 δ 能量形成局部剂量；
局部沉积本身是近似，在界面、横向尾部和 minibeam valley 尤其需要验证。
当前没有可再通过关闭而大幅提速的完整电子 tracking kernel。
经验高斯 δ 横向搬移（sigma = 0.5 mm）已测试并否决（[failed.md](failed.md)）。

仓库另有旧 section-0 delta-tail 搬运、材料电子家族 / 能量包重放候选。
能量包沿记录状态和续接路径搬运已经预算的能量，不让离子再损失第二份能量。
这些候选不在统一生产入口或当前 minibeam 场 YAML 中启用，也不能直接叠加到受限加聚合 δ 模型。
见[能量包输运](include/carbon/electron_packet_transport.hpp)。

中子、光子及衰变产物尚非全部具有完整生产输运链。不支持能量、逃逸和兼容 sinks
分别记账；总能量占比小不代表其局部 halo / valley 剂量一定可忽略。
当前模型不能描述成覆盖全部次级的完整 Geant4 输运。

## 7. 记分与剂量比较

```text
Dose(Gy) = Edep(MeV) × 1.602176634e−13 / voxel_mass(kg)
voxel_mass(kg) = rho(g/cm³) × volume(mm³) × 1e−6
```

使用累计 3D DoseToMedium。IDD 从 3D 记分横向求和得到；异质体素应先按质量将剂量
还原为沉积能量，再形成能量沉积 IDD，不能把裸 Gy 求和当成能量和。
横向 profile、core / halo 宽度分别评价。LET 为独立选项，当前剂量性能对照关闭 LET。

不拟合剂量归一或配准。质量检查包括有限数值、数据来源、抽样 audit、能量记账和零 queue overflow。
overflow 使该片无效，必须拆分重跑。全局能量闭合不证明空间剂量正确，也不等于每个核顶点 Q 值闭合。

### 7.1. CT Gamma

当前 CT Gamma 口径：

- 根据 RTSTRUCT 构造 BODY mask，只评价 BODY 内参考体素中心；记录 ROI、轮廓栅格化 / 插值方法及 mask SHA。
- 按现用 10% 剂量阈值，评价 `BODY ∩ {Dref >= 0.1 Dmax}`，`Dmax` 为参考全体积最大值。
  BODY 归属和剂量阈值是两个独立筛选条件。
- Global 容差以 `Dmax` 为基准，local 以查询点参考剂量为基准；粒子数和几何匹配后评价
  3%/3 mm、2%/2 mm、1%/1 mm、3%/0 mm。
- DTA > 0 时搜索步长为 `DTA/10`，分别 0.3、0.2、0.1 mm，使用三线性插值。
  BODY 限定参考查询点，不把 GPU 的 BODY 外剂量强制置零。格点搜索不等于解析连续最小值。
- DTA = 0 时只比较同体素，不受搜索步长影响。

9 月 5 日无 BODY、固定 0.5 mm 的结果保留为历史证据。

### 7.2. Minibeam 固定 ROI

Minibeam 横向记分为 1000 × 1000 平面（0.1 mm × 0.25 mm）。规范固定 ROI 由网格元数据与节距 3.6 mm 定义：

```text
peak:     |folded x| < 0.25 mm
shoulder: 0.25 mm ≤ |folded x| < 0.9 mm
valley:   0.9 mm ≤ |folded x| ≤ 1.8 mm
```

并限制在中心场 `|x| ≤ 18 mm`。1 mm 深度平面沿深度平滑；Bragg ±2 mm 积分使用未平滑数组。
入射 history 归一化为按 history 数缩放后的绝对 Gy，不用存活粒子数再归一化。

TOPAS `ChargedOriginDoseToMedium` 是来源分类：核带电粒子用自身 Z/A，电子继承已知带电祖先。
GPU charged-origin 图是沉积离子 Z 类，并含浓缩电子。这些可比类不得与尚未匹配的 TOPAS-only
类（`neutral_origin`、`unclassified`）相加。GPU/TOPAS 残差贡献为

```text
Δ_i = (D_GPU,i − D_TOPAS,i) / D_TOPAS,total
```

没有逐 incident-history 的 ROI 矩或独立 shard 时，不确定度标为 unknown。
`|D|/√N_histories` 不是有效的 ROI 误差。TOPAS 多线程 EventID 不是源文件行号。

可选诊断（默认关闭，不得改变生产剂量）：p/d/t/He-4 能区 ROI、原发 C12 ROI 账本、
水入口相空间、水中原发平面。见
[analyze_minibeam_residual_attribution.py](benchmark/carbonminibeam/analyze_minibeam_residual_attribution.py)
与 [minibeamresult.md](minibeamresult.md)。

### 三病例全粒子数比较（2026-09-15 战役）

9 月 15 日三病例 CT 比较仍作为历史战役记录。Gamma 遵循本节 BODY、10% 阈值与 DTA/10 规则，
尚不列未完成的通过率。TOPAS 含 `CarbonIonElasticPhysics`，最新 GPU 生产配置独立核弹性关闭。

## 8. 种类分组与实测吞吐

当前生产算法（2026-09-15）：RT07575 1/20 shard 共 6,481,909 粒子，分两段，已验收候选 wall 68.61 s、程序 elapsed 65.20 s，99.4k histories/s，零 overflow。b1 的 100/200/300 MeV/u 峰值相对误差分别为 −0.0715%、−0.1098%、+0.0424%。患者 BODY Gamma 尚未验收。以下旧吞吐记录与第 9 节方案保留为历史研究；当前 δ 聚合已启用，并配套分步修正。CT Fermi–Eyges 仍为可选研究；对全部 CT C12 强制 0.10 mm 真实 FE 材料刷新步已被否决（[failed.md](failed.md)）。

### 当前显存与分片选择（2026-09-15）

原发按 `history_chunk_size` 分批发射，但次级队列不会在每个原发批次结束时清空：
先完成该 shard 的全部原发，再按代处理次级。队列当前固定容量为 **32,000,000** 条；
次级续跑还为当前代每条轨迹分配 **272 字节状态**及索引/标志。因此 shard 粒子数影响
次级队列及续跑状态的峰值显存，仅看原发阶段约 4.4 GB 的占用会低估需求。

| RT07575 诊断 | 原发数 | 原发批量 | 完整 wall s | 程序 histories/s | 采样显存峰值 MiB |
|---|---:|---:|---:|---:|---:|
| 增大原发批量 | 3,240,955 | 131,072 | 34.65 | 99,216 | 7,817 |
| 增大单片粒子数 | 4,861,226 | 34,816 | 48.88 | 102,989 | 9,517 |

两次均质量通过、零 overflow，使用相同生产物理；显存每 0.25 s 采样，可能漏过更短瞬态。
原约 324 万粒子 / 34,816 批量通常需 34–35 s；这些单次测量不足以给出严格加速置信区间。
增大原发批量没有明确收益，保留生产批量 **34,816**。本次 CT benchmark 将剩余 shard
目标上限调为 **490 万原发**，是任务调度选择，不是通用显存安全上限或物理配置变更。
不同 CT 和能谱仍须检查；显存不足或次级 overflow 的尝试不并入剂量，拆半重跑。
诊断剂量不计入患者总粒子数，重分片保持每个 spot 的整数总数不变。

正式接入回归的 RT07575 6,481,909 粒子为 wall **68.76 s**、程序 elapsed **65.45 s**、
约 **99.0k histories/s**，与上文已验收候选保持一致。

### 默认次级续跑（2026-09-14）

两个生产预设均启用 `secondary_step_chunking: true`，与种类分组一起使用。
每次 kernel 最多执行 16 次完整次级循环，保存存活轨迹并稳定压紧其索引；
队列少于 8192 时直接跑完剩余轨迹，不缩短或合并物理步。
续跑保留能量、位置/方向、RNG 计数、材料缓存、待写回沉积及诊断累计量；
只有真正终止才做末端计分，每代完成后才开始子代。设为 `false` 恢复整条轨迹
单次运行且不分配续跑缓冲；种类分组由自己的开关控制。

9 月 14 日原型每条次级需要 336 字节状态及约 20 字节索引/标志，
实测 288 万次级约增加 1.03 GB 显存；当前状态为上述 272 字节。分配失败会停止运行，应减少每个 shard 的
原发数后合并结果。同时原发已去掉一次重复平均能损查询，只有可选审计需要时
才执行；实际能损抽样保持原样。

已验证原型在 RT07575、100 万原发下：不加独立弹性时 29.5–29.6k histories/s，
相比已消除重复查询的基线提高 36.6–39.7%；含全离子弹性时 28.2–28.6k，
提高 34.7–39.2%。计数、EM 审计、步数一致，零 overflow。
弹性最大剂量差为峰值的 0.000955%（该体素局部约 0.00337%），用户已接受并授权
接入生产。该稳定差异超过自重复波动，原因尚未证明；质量报告记录
`secondary_step_chunking_accepted`。本轮是 GPU 调度比较，没有重新计算 TOPAS Gamma，
原有低密度阈值区及患者 Gamma 验收未完成的说明继续保留。
正式源码开关对照：RT07575 1M、不加独立弹性为 21,559 → 29,535 histories/s
（+37.0%），最大差为峰值的 0.0000966%。

`secondary_species_grouping: true` 在 GPU 上对每代次级建立索引排列，分为 18 种离子
及其他产物共 19 桶。Histogram、prefix sum、scatter 保留粒子记录、parent history 和
RNG stream，子代仍在下一代处理。额外索引内存约为每队列槽 4 字节。

两个生产 YAML 显式开启；配置结构默认 false，以兼容旧配置。日志输出模式及分组时间；
授权接入后质量报告仍保留 `secondary_species_grouping_accuracy_pending`。

性能需分别报告 kernel、程序内部 transport elapsed、完整进程墙钟。
加载、预处理、传输、后处理和输出影响后两者；不能把它们的差值当成某个物理过程的实测耗时。

## 9. 可大幅提速的近似方向——尚未启用

本次文档更新不启用下列近似。目标是在声明误差预算下减少工作量；目前没有达到 50k/s。
分组后的含弹性 1M 实验，原发仍约 37.7 s、次级约 26.2 s，而 50k/s 只允许总耗时 20 s。
因此仅简化次级不可能达到目标，原发输运也必须显著降低成本。

| 候选 | 省略的重复工作 | 潜力与主要限制 |
|---|---|---|
| 联合 EM 块传播 | 多次连续涨落、δ 时钟、散射微步 | 覆盖面最大；联合分布及几何验收困难 |
| 短射程反冲 / 碎片终止核 | 完全终止在同一体素内的很多低能步 | 适用范围较小；先测其实际时间占比 |
| 次级带权 roulette | 只追踪抽中的次级并提高幸存者权重 | 原发 histories/s 增加可能被方差增加抵消 |
| 次级 CSDA / 仅均值快速模式 | 次级 δ 时钟及随机能损细节 | 显式有偏替代模型，不能算完整统一 EM |
| 降低 MCS 准备 / 更新频率 | 重复角度与位移运算 | 改变有限步散射；单独提供数倍收益的依据不足 |

### 9.1. 主攻方向：联合 EM 转移，不是只批量抽 δ

从当前微步模型建立以 `(物种, 材料, 密度, 能量, 块长度)` 为条件的传播模型。
必须**联合**描述受限损失、δ 损失、末端动能、角度 / 位移和块内沉积位置。
stopping 和反应率随动能非线性变化，这些量相互关联；只匹配 δ 均值、方差不够。

传播块不能跳过首次核碰撞或材料边界。核光学深度积分必须与抽样能量轨迹一致，
仅用末端核反应率更新并不等价。路径可能离开后重入体素，不能只检查两端位置。
先限定同质区、远离射程末端和界面的短块，设计明确的回退 / 过渡规则；
不能丢弃所有越界样本后重新抽取“留在体素内”的路径，这会引入条件选择偏差。

Bragg peak、低密度 production-cut 起始区和界面附近先保留原逐步算法。
总损失与块内空间记分需一致，全部能量堆在块首 / 块尾会增加另一种空间近似。
小型联合分布表或降维条件模型可能省掉大量循环，但表体积、查表成本和回退比例也可能抵消收益。
目前没有实测倍数，不承诺达到 50k/s。

与已失败的复合 Poisson / δ 分位数方案的区别是：旧候选仅保留或近似部分联合传播，
同时改变连续涨落和 MCS 的步长。这里不建议原样重跑旧方案。

### 9.2. 较小范围的候选：短射程终止

先针对额外的 EM-only 重反冲，判断剩余输运相对于各体素面的距离和剂量梯度尺度是否足够小。
用一个终止沉积分布替代很多步；只有在声明的更严格范围内才近似为出生点局部沉积。
CSDA range 是平均估计，不是严格射程上界，必须给出射程尾部 / 跨材料逃逸的误差预算。
完整保留残余能量，不将该规则泛化到所有低能 proton / alpha。
对仍可能发生核反应的物种，还要控制被忽略的核碰撞概率。
minibeam 要相对于束宽 / valley 尺度判断，不能只与粗体素尺寸比较。
实施前先按物种和剩余 range 分解步数 / 时间；目前不能断言该项占据大部分耗时。

### 9.3. 带权抽样与显式简化次级 EM

roulette 存活概率为 `p` 时，幸存粒子权重改为 `w/p`。只有所有子代、dose / LET、
逃逸和能量账本正确传递权重，剂量期望才保持。当前队列 / 记分存在单位权重路径，
不是加一个开关就能实现；逐 history 实现能量闭合和估计量记账需要重新设计。
真实 overflow / 丢失检查必须与 roulette 的统计涨落分开。

用固定统计不确定度所需时间评价，例如 `1/(time × variance)`，不能只看原发 histories/s。
稀有碎片的相关丢弃可能恶化 local Gamma、halo 和 valley 的统计精度。
次级仅均值模型则引入系统偏差，必须单列为近似模式，不能继续声称完整统一 EM。
整体关闭次级涨落、核弹性或中性剂量不适合作为 minibeam / valley 默认模型。

### 9.4. 不重复的路线与验收顺序

- RNG dummy 的加速不是物理抽样加速。已撤回的 buffer / macro-tick 不属于生产选项，
  secOFF / K 的旧速度不能作为完整统一 EM 基准。
- 逐电子 Poisson 批处理已测得更慢。只处理 δ 的分位数候选曾在单轮 RT07575 提速约
  10.9%，但未通过 peak / R80 筛选，不属于保精度替代方案。
  见[批处理](docs/delta_batch_sampling.md)和[分位数结果](docs/rt07575_quantile_optimization.md)。
- 当前电子已局部沉积、核末态已查表重放；关闭并未运行的完整电子 tracking、
  或“替代在线核级联”，都不能在当前实现中再省出相应成本。
- 在把入口、C12 输运、能损分区和空间计分拆开之前，不得用改 primary MCS 或启用电子包
  去补偿 minibeam valley 残差（第 11 节）。

先测可覆盖区域 / 物种的时间和步数。若可优化部分占总时间 `f`、自身加速 `s`，
不计新增开销的总加速上限为 `1 / [(1−f) + f/s]`。
随后验证条件损失分布及相关性、射程 / 末端状态分布和能量记账。
使用默认 TOPAS 参数验证 100/200/300 MeV/u、b3/b4 界面及 RT07575，350/400 作补充。
同粒子数、多独立 seed 比较 IDD peak / R80、core / halo、横向 profile 和 BODY 内 local / global Gamma。
近似算法通常不再要求事件计数完全相同，应评价统计一致性和剂量偏差，而非套用调度重排的逐计数等价门槛。

可暂沿用此前“峰误差增加 ≤0.3 个百分点、R80 位移 ≤0.1 mm”作为继续投入的筛选，
它们不是已成立的临床验收标准；正式比较前还需预先声明允许的 Gamma 降幅及统计不确定度。
扩展 minibeam 时增加 valley 剂量 / PVDR 和空间尾部验证，不能调整参考 TOPAS 去追随近似模型。

## 10. 复现与数据

正式 EM 还需 `data/em/unified_em_delta_moments_v2.bin`。安装核心数据后运行 `python3 tools/build_delta_moments.py`，再运行 `python3 tools/verify_unified_em_data.py`。矩表由固定哈希核心包生成，包含全部材料/离子节点；旧 Release 不含该派生表。

按 C++20 / SYCL、`nvptx64-nvidia-cuda`、本地 `sm_75` 构建，冻结 executable、
解析后 YAML、数据 SHA、CT / 源变换、histories / spot 分配、seed、记分与分片 manifest。
除 clone 外还需大物理二进制及外部 CT / 源输入，不能从旧 Release 标签推断当前所需数据已齐备。

Schneider CT 每次运行前执行 `python3 tools/verify_schneider_v2_1_data.py`。
精确的 v2.1 核 / stopping 栈仍是最低要求，不允许水 / 四分类或旧 schema 回退。
统一 EM 使用 `data/em/unified_em_v1.bin`，SHA256：
`8c5d970b3b639bfca2f448730271bed4fc04721aba73100e2efbe09dffe44855`。
未来传播 / 反冲近似表属于新候选数据，不自动继承该包的授权或验证。

GPU 只在本地。未指定 TOPAS 环境时使用本地 `sbatch`，数据放 `/mnt/sda/wuwei`；
明确指定远程 CPU 主机 / 集群时按仓库规则允许提交。所有任务合计最多 192 CPU 线程、160 GB 内存。
大任务按需分片，任何 overflow 片必须拆小重跑，只合并逐片接受的结果。

[extensions 套件](extensions/README.md) 用于对照 TOPAS 4.2.p3 / Geant4 11.3.2 重新提取
EM、CINEL03、Schneider 率、全离子弹性和 Copper 包。不同 TOPAS/Geant4 版本就是新包版本。
大型 `.bin`/`.cinpkg` 不进 Git，哈希记在 `extensions/package_manifest.json`。

本次把文档更新到 2026-09-20 工作树，没有新增 CT Gamma 或 minibeam valley 匹配的物理验收。

## 11. Minibeam 验证口径（当前）

Minibeam 开发冻结原发水能损 0.9958、Copper/缝几何、package 和 seed，除非受控 A/B 点名改动。
次级 C12 FE 和次级 0.9958 不得作为生产结论开启；p/d/t/He-4 水 FE 是场 YAML 实验选项，
不是已完成的 CT 标定。

同源 C12 水输运使用同一份 parent-0 C12 入口集合，在 GPU 与 TOPAS 两侧比较。
带空 history 与不带空 history 的 TOPAS 相空间运行不可互换：250 MeV/u 256k 的
`topas_6363`（empty on，256000 histories）不能与 30878 粒子的 GPU 回放比较，
而 `topas_6364`（empty off，发射 30878，weight 1，`PhaseSpaceMultipleUse=1`）可以。
不得机械地按存活/入射比缩放剂量。同一 GPU 水核上 GPU 入口对 TOPAS 入口是
**入口替换敏感性**，不是跨引擎同源比较。

在该同源口径下，250 MeV/u 全物理 40 mm valley 的 GPU/TOPAS 为 0.9867，
300 MeV/u EM-only Bragg valley 为 1.0037；EM-only 对在 40 mm 与 160 mm 的 C12
穿越、能量和角度接近。全链 valley 残差更大，不能按占比归因到 primary MCS。
下一处 kernel 修改必须先把入口、C12 输运、能损演化和计分位置拆开
（[minibeamresult.md](minibeamresult.md) 第 37 节）。

## 参考

- [FRED carbon 文献分析](docs/FRED_Carbon_Fragmentation_Model.md)
- [统一 EM 数据与模型](docs/physics/unified_em_v1.md)
- [旧 primary-water 联合模型，已从代码移除](docs/physics/water_joint_em_v1.md)
- [历史物理规格](docs/TOPAS_GPU_Physics_Model.md)
- [宽束 Fermi–Eyges](docs/broad_beam_fermi_eyges.md)
- [记分契约](docs/scoring_validation.md)与[旧稿归档](docs/archive/README.md)
- [Minibeam 运行记录](minibeamresult.md)
- [已否决路线](failed.md)
- [TOPAS 提取套件](extensions/README.md)
