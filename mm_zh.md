# 材料与方法

更新：2026-10-05；实现基线 `ebff981`。[English](mm.md)。本文描述当前八个预设的物理过程、近似、评分及执行方式，详见[清理验证](docs/repository_cleanup_20261005.md)与[CT 吞吐和验收记录](docs/ct_throughput_20261005.md)。另见[CT broadbeam 电子优化](docs/broadbeam_300k_20261005.md)与[最新 EM 搜索复用](docs/broadbeam_em_search_20261005.md)。两个 CT broadbeam 已通过指定门槛；肺 CT proton minibeam 仍未通过。历史测量只对应其记录的程序、束流、几何及统计量。

## 八个预设

仅保留以下八个 `config/` 入口。粒子种类和几何各自选择，共用电子传播实现。

| 预设 | 离子多重散射 | 电子出生选区 | 当前验证状态 |
|---|---|---|---|
| `proton_ct_broadbeam.yaml` | Gaussian Fermi–Eyges | CT 低密度区及周围5 mm | 当前三维 local 1%/1 mm 99.8935%，通过 |
| `carbon_ct_broadbeam.yaml` | Gaussian Fermi–Eyges | CT 低密度区及周围5 mm | 当前三维 local 1%/1 mm 99.8699%，通过 |
| `proton_ct_minibeam.yaml` | Urban | CT 内全部 primary 电离损失 | 共用 Urban/FP32；肺 CT 尚未通过新 >99% 门槛 |
| `carbon_ct_minibeam.yaml` | Urban | CT 内全部 primary 电离损失 | 调度提速1.946倍；输运审计通过；剂量待验收 |
| `proton_water_broadbeam.yaml` | Gaussian Fermi–Eyges | 单位密度均匀水中选区为空 | 两批 smoke 通过；water 专项验证推迟 |
| `carbon_water_broadbeam.yaml` | Gaussian Fermi–Eyges | 单位密度均匀水中选区为空 | 两批 smoke 通过；water 专项验证推迟 |
| `proton_water_minibeam.yaml` | Urban | 水内全部 primary 电离损失 | 共用 worker smoke 通过；剂量待验收 |
| `carbon_water_minibeam.yaml` | Urban | 水内全部 primary 电离损失 | 共用 worker smoke 通过；剂量待验收 |

两个 CT broadbeam 预设使用 RT07575 计划输入。两个 CT minibeam 预设使用肺 CT，分别为170 MeV 质子和250 MeV/u C12；它们不是 RT07575 minibeam full plan。计算后者需要使用该计划的实际 CT、spots、归一化和参考剂量。Carbon water minibeam 保留250 MeV/u spots 源及1.2%相对能散。

## 材料、源和输运几何

MAIGO 是 C++20/SYCL 蒙特卡罗程序，将带电粒子凝聚输运与 TOPAS/Geant4 提取的物理表、条件终态结合。提取/参考环境为 TOPAS 4.2.p3、Geant4 11.3.2。GPU 上没有运行完整 Geant4 physics list：有限响应库、插值、粒子覆盖范围及下述近似都是方法的一部分，验证结论只适用于指定配置和参考。

CT 输入提供体素尺寸、原点、密度和 Schneider 组成编号。输运采用当前位置的组成及真实密度，不把全部 CT 体素视为按密度缩放的水。离子 EM 表包含175个材料/密度条目、18种带电粒子；电子使用另一套82节点 Schneider 注册表，两者离散方式不同。离子注册种类为 p、d、t、He3/4/6、Li6/7、Be6/7/9/10、B8/10/11、C10/11/12；额外弹性反冲核需要专用阻止本领数据。见 [CT 几何](include/carbon/ct_grid.hpp)、[带电粒子注册表](include/carbon/charged_species.hpp)、[EM 提取](extensions/tools/unified_em/prepare.py)。

水的阻止本领参考为 **TOPAS Water_75eV，平均激发能75 eV**。`data/water_unified/g4_water_material.json` 提供组成/辐射长度信息，其中的 G4_WATER 标识不把阻止本领参考改为78 eV。`minibeam_water_primary_stopping_power_scale` 保持1.0。电子材料 section25 表示水；离子 EM 注册表用 section−1 表示水，两者不可混用。

源采用配置的平坦照射野或 spot 列表。TPS 源按 spot 权重分配历史数，抽样所配置的能散及横向位置—角度相关分布，并根据计划姿态与 isocenter 将束流坐标转换到患者坐标。`mu` 与 `histories` 权重模式的分配规则不同，仅改变 `number_of_histories` 不会缩放按历史数加权的计划。以 MeV/u 指定的碳离子源能量在输运时转换为总动能。所选预设可以加入上游空气能损及源协方差修正，但这不等于显式模拟上游电子簇射。见 [TPS 源](src/tps_source.cpp)。

CT 材料面限制输运步长，包括斜入射。评分几何与物理几何分离：离开评分 ROI 不一定离开输运世界。水中电子 worker 使用覆盖物理水箱的单个 Water_75eV 体素，评分网格可以更细；这不会启用水中的 Schneider 离子物理。跨界后更新实际材料和密度，数值边界微移用于导航，不代表额外的物理能损。

## 离子电磁过程

### 平均电离能损与射程

共用 EM 包为各受支持粒子/材料提供受限阻止本领、射程/反射程、有效电荷与涨落参数，以及产生阈值以上的 delta 电子能损矩。原生样条数据重建阻止本领/射程，同一组成内在相邻密度节点间插值。小能损采用阻止本领乘以接受的**真实路径长度**；较大能损采用射程反演及提取的离子修正。达到射程终点时沉积剩余能量。这是凝聚输运，不逐个抽样原子激发。

总电离损失等于受限连续部分与聚合的阈值以上 delta 部分，两者总和受剩余动能限制。Delta 均值包含局部能量依赖修正。离子扣除能量与电子空间沉积是两个操作：扣除的能量随后由出生响应分配，单次 delta 抽样不会直接生成一组物理电子轨迹。见[平均能损](include/carbon/primary_em_mean_candidate.hpp)、[运行时 EM 状态及能损](include/carbon/unified_em_view.hpp)。

### 能损涨落

开启 straggling 时，受限能损使用提取的 ion/universal 涨落模型及其材料产生阈值、色散和运动学最大能量转移。薄/厚步长采用不同抽样区间，并非统一高斯近似。独立的聚合 delta 部分使用矩匹配 Gamma 分布，shape=`mean²/variance`、scale=`variance/mean`；方差为零时返回均值。它匹配两个聚合矩，不保留精确 compound-Poisson 的零碰撞概率和高阶矩。见[受限涨落抽样](include/carbon/restricted_fluctuation_candidate.hpp)、[delta 抽样](include/carbon/delta_loss_sampling.hpp)。

**Carbon CT broadbeam 当前关闭 primary 和 secondary ion 的能损涨落**，其余七个预设开启。关闭离子涨落后使用平均能损，但核反应、散射、电子出生及电子 worker 自身的能损涨落仍然是随机过程。这一预设差异属于当前已验证配置。

### 离子多重库仑散射

Broadbeam 使用 Gaussian Fermi–Eyges 分支。在散射功率近似恒定的小段内，单个投影方向的角度/横向协方差为 `A0 = T L`、`A1 = T L²/2`、`A2 = T L³/3`，相关正态抽样同时产生角偏转与横向位移。散射功率取决于电荷、动量、速度及当地材料辐射长度。粒子/能量系数来自水板拟合，未单独参数化的粒子采用共用系数。Gaussian 分支将拟合尾部贡献并入二阶矩，不再单独抽样 Poisson 尾部。这些拟合是近似，不是原生 Urban 散射；历史 G4_WATER 水板来源指散射拟合，不替代 Water_75eV 阻止本领参考。见 [Fermi–Eyges 实现](src/detail/sycl_device_math.inc)。

Minibeam 使用共用的**离子 Urban v2** 路径。材料相关原生参数限制散射/路径，几何限制几何推进长度，能损使用相应接受的真实路径；步长接受后才提交角度和横向位移抽样。Primary 与受支持带电碎片使用同一框架及各自粒子数据。最大步长是上限，不是每一步固定走的距离。两个 CT minibeam 均启用 `urban_ct_density_scaling`，按实际/参考密度缩放原生材料上下文中的路径、阻止本领/射程及平均自由程处理。该固定 cut 近似不包含全部密度效应及产生阈值能量变化，因此共用 Urban 代码本身不代表与原生实际密度模型严格等价。见 [Urban 集成](src/detail/sycl_urban_global.inc)、[Urban 数据包](include/carbon/urban_mcs_package.hpp)。

### 各预设实际生效的设置

下表按[配置解析](src/config.cpp)展开 `transport_profile` 后的设置整理，不是简单罗列 YAML 原始键。离子动能截止为0.1 MeV **总动能**，不是0.1 MeV/u；电子阈值独立并随材料变化。

| 预设 | 离子最大步长，mm | 离子多重散射 | Primary / secondary ion 涨落 | 独立 all-ion 核弹性包 | 中子 worker |
|---|---:|---|---|---|---|
| Proton CT broadbeam | 0.5 | Gaussian FE | 开 / 开 | 开 | 关 |
| Carbon CT broadbeam | 0.5 | Gaussian FE | 关 / 关 | 关 | 关 |
| Proton water broadbeam | 0.25 | Gaussian FE | 开 / 开 | 开 | 关 |
| Carbon water broadbeam | 0.25 | Gaussian FE | 开 / 开 | 关 | 关 |
| Proton CT minibeam | 0.25 | Urban v2 | 开 / 开 | 开 | 关 |
| Carbon CT minibeam | 0.25 | Urban v2 | 开 / 开 | 开 | 关 |
| Proton water minibeam | 0.25 | Urban v2 | 开 / 开 | 开 | 开，实验性 |
| Carbon water minibeam | 0.005 | Urban v2 | 开 / 开 | 开 | 关 |

CT broadbeam 的平均能损步长比例限制为质子0.02、碳0.05；water broadbeam 使用默认0.01。边界、射程与相互作用还会缩短步长。全部 minibeam primary 步长≤0.25 mm，其中 carbon water 保留更严格的0.005 mm。因此，共用算法不表示八个预设采用完全相同的数值设置。

## 核反应和带电次级粒子

### 离散核弹性散射

配置 `all_ion_elastic_file` 时，材料/粒子相关弹性反应率及抽样角分布更新入射粒子能量和方向，并产生靶核反冲。带电粒子注册表或专用反冲阻止本领表覆盖的反冲核，在阈值以上进入队列；阈值以下动能局部沉积。未覆盖的阈值以上反冲核进入未输运能量账目。队列溢出使该次运行不合格，不能默许丢失能量。

该离散核反应与电磁多重散射不同。特别是，单独看到 `enable_nuclear_elastic: false` **不能**判断 独立配置的 all-ion 包已关闭。两个 carbon broadbeam 不加载该包；全部 proton 预设与两个 carbon minibeam 加载。见[弹性数据及抽样](include/carbon/all_ion_elastic.hpp)、[主/次级输运分派](src/transport_sycl.cpp)。

### 非弹性反应与碎片级联

宏观反应率取决于入射粒子、能量、局部组成和密度。当前 Schneider v3 路径将元素分反应率限制在支持的通道范围内，沿能量插值再求和；靶元素抽样使用与相互作用率相同的分反应率。缺失通道不替换为氧或邻近元素。见[靶元素/反应率抽样](include/carbon/schneider_target_sampler.hpp)。

CINEL03 按 `(projectile Z, A, target Z, collision energy)` 保存相关的原生核反应事件。覆盖范围内按线性概率选择相邻能量节点之一，再在该节点抽一个事件；允许的节点间隙最多5 MeV/u。**离子 CINEL03 路径不逐事件缩放终态动能。** 整个终态共用一个随机方位角旋转到入射坐标系，保留事件内方向关联。有限节点造成的碰撞能量差异及核反应剩余能量账目保持可见，不把碎片独立地按边缘谱重新生成。见[查表约定](include/carbon/inelastic_package_v3.hpp)、[事件旋转](include/carbon/cinel03_event_rotation.hpp)。

数据来源记录为：proton primary 的 Binary Cascade（`QGSP_BIC_HP`）、C12 primary 的 `FTFP_INCLXX`、secondary ion 的 `g4em-standard_opt4 + g4ion-binarycascade`。这些名称描述离线提取来源，不表示 GPU 正在执行一个完整、统一的同名 physics list。活跃文件见[数据包清单](docs/package_manifest.json)。

过程局部沉积在碰撞点评分。覆盖的阈值以上带电产物进入次级队列，继续 EM 输运、散射及符合条件的后续核反应；阈值以下带电能量局部沉积。当前患者/水中 CINEL03 分派在反应后终止入射轨迹，输运记录的直接产物；数据中保存的母粒子字段不代表已实现通用母粒子续跑。患者/水中 secondary inelastic generation 上限为2，铜中碎片上限为3。达到代数上限仅停止新的非弹性分支，不终止现有带电粒子的剩余 EM 输运。对次级核反应范围外粒子所选的 `em_only` 策略，以及有限同位素/事件覆盖，都是明确的模型限制。

中性产物、未覆盖产物、逃逸和核反应剩余量分别有审计字段；中性/未覆盖分项不能与已包含它们的 combined nuclear sink 重复相加。核反应光子不会自动送入电子响应的 photon worker。关闭中子模块时，核反应中子能量计入未输运量，不计为局部剂量。

### 铜准直器与上游输运

Minibeam 在进入模体前穿过配置的铜狭缝和空气间隙。铜密度8.96 g/cm³，使用自身阻止本领、Urban 散射、弹性和非弹性数据，因此准直器衰减不只是几何遮挡。铜中带电步长上限0.05 mm。铜内 proton 能损选项在 proton 预设为 `g4_universal_delta`，在 carbon 预设保留 `legacy_clipped_gaussian`。铜碎片涨落除 carbon water minibeam 外均开启。患者/水中 primary 出生包不提供上游电子簇射。具体铜设置保留在各 [minibeam YAML](config/) 中。

### 可选中子输运

仅 proton water minibeam 默认启用实验性中子 worker。在水/铜/空气世界中，核反应出生中子使用弹性、非弹性、俘获、裂变表，并按边界处理光学厚度飞行。满足条件且能量高于0.1 MeV的独占弹性事件在实际能量下使用两体运动学；其余库事件保留相关终态，同时缩放能量并记录 kinetic gain。带电反冲使用阻止本领数据；此预设关闭可选的水中反冲 Urban 输运。中子产生的光子/电子仍未解析，单独记账。它不是已验证的通用 CT 中子求解器。见[中子实现](src/neutron_transport_sycl.cpp)。

## 共用电子和光子输运

### 出生分配与能量包估计量

Primary 电离损失按粒子、材料、密度及能量对应的能量加权出生比例分配：剩余部分局部沉积，代表电子的能量进入任务。Proton 与 C12 使用不同出生包，后续电子/光子物理共用。当前 primary-birth 实现不额外生成 secondary ion 的显式电子出生，也不提供铜/空气中的上游电子 cascade。核反冲沉积不混入 primary 出生响应。

任务携带**代表能量 W**和抽到的物理电子动能 **E**，二者不能混用。Group32 用加权 reservoir 将出生贡献合并为代表任务，不等于输运32个 analog 电子。出生能量和极角联合抽样，方位角相对于离子方向抽样。见[出生分布](include/carbon/electron_birth_bank.hpp)、[任务状态](include/carbon/electron_birth_task.hpp)。

Worker 沿级联的一条能量加权分支推进。软能损为 ΔE 时，标准估计量以 ΔE/E 的概率在抽样位置沉积 W 并结束；否则以物理能量 E−ΔE、原权重 W 继续。硬碰撞的能量加权条件 CDF 选择局部沉积、继续电子、次级电子或光子，也保留 W。这样无需保存整个簇射即可估计提取响应下的期望剂量，但不复现事件多重性或全部簇射关联。分组、轮盘抽样改变方差；有限响应统计及插值也可能影响准确性。见[能量包估计量](include/carbon/electron_charged_packet.hpp)、[FP32 worker](src/detail/sycl_independent_electron_fp32.inc)。

### 独立带电电子步进

Worker 在当前材料和能量下求受限阻止本领、射程、弹性率及硬电离/轫致辐射率。普通步长受 **0.05 mm**、**0.005平均能损比例**、剩余射程、下一个物理体素面及候选相互作用距离共同限制。这些电子设置与离子步长上限不同。材料产生阈值和低能端点来自算子，不使用离子的0.1 MeV截止。

软能损使用提取的电子 Urban 涨落参数；弹性角偏转采用含 Mott 修正的 Geant4 GS 单次弹性散射律。这既不是离子 Urban 多重散射，也不是新提出的低密度凝聚 MCS 近似。硬事件采用已检查的碰撞率上界生成候选，再按实际率接受/拒绝为空碰撞，并从提取的电离/轫致辐射条件终态抽样。跨界后从当前状态在新材料继续；缺覆盖和迭代上限失败会被报告，不转换为成功的局部沉积。见[步进算法](include/carbon/electron_charged_step.hpp)、[弹性散射律](include/carbon/electron_elastic.hpp)、[Urban 能损](include/carbon/electron_urban_fluctuation.hpp)。

同一组成内，质量阻止本领/射程及质量反应率在实际密度节点之间插值，再用物理密度转换回来。硬碰撞终态的节点混合按过程反应率加权。带电模型采用较低密度节点的弹性角律与该组成的 Urban 常数；该选择及有限密度插值属于近似，不表示所有密度响应完全相同。见[密度模型](include/carbon/electron_charged_density.hpp)。

### Joint-response 加速

Schneider 覆盖状态可使用紧凑相关 joint 算子推进短段。必须在**抽样之前**确认全部可能位移范围都位于相同组成、**相同密度**的区域内，不能按抽出的结果决定是否接受该近似。每种组成使用最低索引密度作参考，按 `参考密度 / 实际密度` 缩放长度和空间支持，不将支持范围扩大到参考范围之外。带电段仍≤0.25 mm。靠近界面或超出 joint 支持时，同一个包转入独立带电 worker；光子后继在实际几何中输运，不重放源球中的光子飞行。

出生、光子转电子以及带电推进的周期检查点会尝试 joint。Water_75eV 没有 joint 算子，使用独立步进。Joint 密度缩放与有限样本算子仍需剂量验证，不是严格等价替换。运行时不回退到已删除的旧空间轨迹包。见 [joint 算子](include/carbon/electron_joint_pilot.hpp)。

### 光子分支

共用光子包包含光电吸收、康普顿散射和瑞利散射；当前亚 MeV 覆盖不包含电子对产生。抽取光学厚度 `τ = −ln(U)`，在实际 CT/水体素序列中逐段扣除 `μ × 距离`，直至碰撞或物理逃逸。材料界面本身不沉积能量，也不重抽剩余光学厚度。光子不受带电粒子0.25 mm步长限制。

碰撞时先按反应率选择过程，再按能量加权条件终态选择局部沉积、继续光子或交回带电 worker 的电子。质量衰减系数按物理密度缩放；条件终态在密度相邻节点间按几何插值权重混合，这是明确的有限节点近似。能量在抽样的相互作用/沉积位置评分，不沿光子飞行路径摊开。见[光学厚度输运](include/carbon/photon_optical_transport.hpp)、[光子包抽样](include/carbon/photon_response_bank.hpp)。

### Broadbeam 选区和统计加速

四个 minibeam 在整个模体启用 primary 电子出生；broadbeam 使用：

```yaml
broadbeam_electron_tracking: true
broadbeam_electron_density_threshold: 0.1
broadbeam_electron_halo_mm: 5.0
```

CT 对密度≤0.1 g/cm³ 区域做5 mm **盒形扩张**，按实际抽到的出生位置决定是否入队。选区外电子贡献局部沉积，入队后可以离开选区继续输运。单位密度均匀水的选区为空，因此 water broadbeam 在此策略下保持局部电子沉积。Tracking OFF 不加载独立电子包，无需 proton 专用空间 tail 或 carbon 专用 replay 路径。

两个 CT broadbeam 使用 group32、按出生能量/密度排序的任务索引，以及低密度和选区内组织各0.5的存活概率。幸存包赋予 W/p，继续电子/光子输运；被拒绝包计入带符号轮盘账目。它保留现有估计量的期望并改变方差。这些排序/轮盘选项不应用于 minibeam 或 water。体外空气电子没有被直接丢弃或强制局部沉积；研究性的期望值评分与低密度凝聚散射方案也不是当前预设基线。

## 剂量评分、数值与执行

离子局部能损、核反应局部沉积及阈值以下带电能量累加到对应评分体素；电子包在抽样沉积位置评分。同一体素的连续贡献可以先缓存再做原子更新。输出为按实际 CT 体素质量或水质量计算的 **dose to medium**，不是用阻止本领换算出的 dose to water：

```text
mass_kg = density_g_cm3 × voxel_volume_mm3 × 1e−6
D_Gy = accumulated_energy_MeV × dose_output_scale × 1.602176634e−13 / mass_kg
```

CT 评分器须匹配 CT 网格，计算质量时密度下限为1e−6 g/cm³。深度聚合量不能替代逐体素按质量计算的 CT 剂量。原始能量/剂量是历史累加值，剂量写出程序不会自动除以历史数；比较每 primary 或 full-plan 剂量时，须明确使用记录的 spot 分配、总历史数及 `dose_output_scale`。见[输出与审计](src/io.cpp)。

GPU 体素评分/原子累加使用 **FP32**（`CARBON_DOSE_FP32=ON`、`CARBON_DOSE_FP64=OFF`）。生产路径的独立电子/光子采用 FP32 输运和几何，CDF 保留整数抽样。这不表示所有主机计算、磁盘数据表、离子局部累加器或审计都为 FP32；参考 helper 中出现 `double` 也不代表选中的电子 kernel 使用 FP64。

Primary/带电离子采用计数器式 Philox4x32-10，以历史、步号和抽样维度标识随机流；次级粒子身份不依赖原子分配的队列槽位。电子 worker 使用任务自身的64位线性同余状态与整数 CDF 抽样；中子模块使用独立 xorshift 流。因此，任务顺序可改变 FP32 原子累加顺序，而不改变任务自身的输运随机流。见[离子 RNG](include/carbon/rng.hpp)、[电子 FP32 实现](include/carbon/electron_fp32.hpp)、[中子 worker](src/neutron_transport_sycl.cpp)。

Primary、带电次级和电子工作使用有容量限制的队列/分块。粒子分类和固定步数的次级续跑改善调度；单进程连续批次复用不可变数据包和几何。最新 EM 优化在加载时检查能量/阻止本领/射程搜索网格一致性，同一步在密度相邻端点复用三个区间索引；保留插值数值与随机抽样，反射程和硬碰撞率搜索仍分开，网格不同则独立搜索。该实现优化与响应插值、FE 散射、有限核事件库及电子能量包近似性质不同。

运行合格需要同时通过完成数、EM/输运覆盖、有限状态、能量/轮盘账目、零队列溢出及剂量检查。未覆盖能量、有限级联代数和未输运核反应中性产物必须保持可见。单一几何的 gamma 通过不代表省略通道、超覆盖能量或全部八个预设都已验证。

## 数据包

电子包注册表见 [docs/electron_packages_manifest.json](docs/electron_packages_manifest.json)，共三个包：

| 包 | 运行内容 | Payload 字节数 |
|---|---|---:|
| `data/electrons/proton/` | 质子 `birth_bank.bin` | 328,620,496 |
| `data/electrons/carbon/` | C12 `birth_bank.bin` | 1,033,490,328 |
| `data/electrons/shared/` | `joint.bin`、charged/photon operators 与索引 | 3,837,722,550 |

去重后的 payload 共5,199,833,374字节，不含 JSON 清单。每次只加载所选粒子的出生包和共用包。83节点加载器报告的设备数据约为质子3.127 GB、碳3.597 GB，不包括其他物理表、队列、评分器，也不代表进程总占用或加载峰值。

Birth、charged、photon 各含82个 Schneider 密度节点和1个 Water_75eV 节点。电子材料 section25 表示水，Schneider 仍为0–24。水中电子/光子覆盖至0.8 MeV；Schneider 保持0.65 MeV。水中 primary 出生数据覆盖质子250 MeV、碳300 MeV/u；Schneider 分别为250、255 MeV/u。不能把这些上限解释为任意源均已验证。实际材料/能量覆盖缺失必须报错，不能替换成水或局部沉积。

现用格式为 MAIBIR02 birth、MAIJNB02 joint、MAIECB02 charged index、MAIPHB02 photon index。索引为材料编号、密度、相对文件路径三列。检查内容包括尺寸、字节数、材料/粒子身份、有限值、抽样支持及碰撞率上界。运行时和当前提取工具均不再做文件摘要校验。旧配置键直接报错，不提供摘要字段或旧包格式的兼容入口。

其他必要包包括统一 EM/delta moments、粒子特定核反应率/事件、所选弹性/反冲数据、minibeam Urban 表、铜/空气和可选 neutron 数据。活跃依赖见 [docs/package_manifest.json](docs/package_manifest.json)。大文件和患者输入保存在被 Git 忽略的 `data/`、`inputs/` 中，仅克隆仓库不足以运行全部预设。

## 编译和运行

需要支持 CUDA 的 SYCL 编译器及插件。RTX2080Ti 示例：

```bash
cmake -S . -B build \
  -DCMAKE_CXX_COMPILER=/home/wuwei/sycl_workspace/llvm/build/install/bin/icpx \
  -DCMAKE_BUILD_TYPE=Release -DCARBON_ENABLE_SYCL=ON \
  -DCARBON_ENABLE_MINIBEAM=ON \
  -DCARBON_SYCL_TARGETS=nvptx64-nvidia-cuda -DCARBON_CUDA_ARCH=sm_75 \
  -DCARBON_DOSE_FP32=ON -DCARBON_DOSE_FP64=OFF
cmake --build build -j 6
python3 scripts/check_electron_packages.py
```

其他 GPU 应选择相应架构。最新已测实现提交为 `ebff981`，冻结程序为 `out/broadbeam_ions_20261005/shared_v2_carbon_mc`，已包含更早的次级粒子弹性检查修复（`5cc096d`）。仅修复检查的旧程序和清理前程序是历史对照，不是最新基线。本地 `out/` 程序不随 Git 发布，复现时应编译对应源码并保留实际生效配置。

单进程连续批次：

```bash
build/carbon_mc --config config/proton_ct_minibeam.yaml \
  --batch-count 10 --histories-per-batch 1250000 \
  --batch-output out/proton_ct_minibeam_run
```

输出目录应为新目录。每批独立种子、剂量和质量报告，同一进程复用数据包及只读几何。计划权重按请求的总历史数分配，已归一化的计划剂量不能再次乘以批次数。不同配置依次运行可用 `--plan-manifest FILE`，并用 `--plan-preflight` 预检查。

## 验收和吞吐优化

DTA 非零时，参考点 r 的 local gamma 为评价位置 x 上 `sqrt((|x−r|/DTA)² + ((D_eval(x)−D_ref(r))/(δ × D_ref(r)))²)` 的最小值，δ 为剂量差异比例。通过率统计符合评价条件的参考点中 gamma≤1 的比例。当前 broadbeam 使用搜索球内的笛卡尔候选偏移；历史 minibeam 使用的径向评价器在下文单列。搜索步距为0.1 × DTA。

四个 CT 预设采用以下验收标准：

| CT 预设 | 剂量门槛 | Gamma 搜索步长 |
|---|---|---|
| Proton / carbon broadbeam | Local 1%/1 mm **>99%** | 0.1 mm |
| Proton minibeam | Local 3%/0.2 mm **>99%** | 0.02 mm |
| Carbon minibeam | 暂不以剂量准确性为门槛 | 若计算则单独报告 |

对比前固定参考、绝对剂量归一化、坐标映射、剂量阈值和评价 mask，并随结果报告。当前 RT07575 broadbeam 使用参考最大剂量10%以上的全三维评价区，未拟合剂量比例或配准。搜索步长变化不允许伴随评价区或分母改变。零 DTA gamma 保留为受蒙卡噪声影响的诊断。

RTX2080Ti 最新 broadbeam 结果采用两个独立种子的连续吞吐及合并剂量 gamma；minibeam 行保留此前较大统计量研究：

| 预设 | 历史数 | 连续批次 primary/s | TOPAS local gamma |
|---|---:|---:|---|
| Proton CT broadbeam | 2 × 12,531,500 | 203,021 | 1%/1 mm：**99.8935%**，通过 |
| Carbon CT broadbeam | 2 × 6,481,909 | 138,019 | 1%/1 mm：**99.8699%**，通过 |
| Proton CT minibeam | 20,000,000 | 51,821 | 二维投影 3%/0.2 mm：**98.3183%**，未通过 |
| Carbon CT minibeam | 两个种子，每个对照/优化组各3 × 200,000 | **10,048**，对照5,165 | 剂量准确性推迟验收 |

最新同轮对照中，质子201,356→203,021/s（+0.8%，变化较小），碳离子129,244→138,019/s（+6.8%）。两种粒子的合并 gamma 均与同种子对照完全一致。共用 EM 模块复用同一步中相邻密度节点的搜索下标，保留原插值、能损、随机数及输运；网格一致性只在不可变数据包加载时检查一次。详见[EM 搜索复用报告](docs/broadbeam_em_search_20261005.md)及[数据](docs/broadbeam_em_search_20261005.json)。

[上一轮电子优化](docs/broadbeam_300k_20261005.md)已接入出生排序及组织存活率0.5，得到+22.8%/+22.0%的连续吞吐改善及 BODY 统计效率改善，本轮保留其估计量。两轮的种子和计时条件不同，不能直接比较绝对吞吐或 gamma 推断额外收益。300k/s仍未达到。表中 minibeam 保留此前较大统计量研究；最新三批一致性测试为质子49,147→50,524/s、碳离子9,965→10,429/s，不构成新的 gamma 验收。

Proton minibeam 沿用历史 PyMedPhys 0.41.0 径向搜索（`interp_fraction=10`），比较肺 CT 的 BODY 投影剂量：|y|<15 mm、|x|<15 mm、参考最大值3%以上。它不是三维体素或 RT07575 full plan 验收。笛卡尔格点交叉检查在同样0.02 mm步距下为97.1013%，两种算法均未通过 >99%。PyMedPhys 峰/谷条带分别为99.9605%和99.0052%，不能替代全场门槛。共用 EM 搜索优化没有改变该剂量结论；本轮 minibeam 检查为实现一致性回归。

Carbon CT minibeam 的 `history_chunk_size`、`material_electron_queue_chunk` 改为131,072，并开启已有的按粒子分组及16步次级粒子续跑。Group32、16,384个电子 worker、576 MiB队列预算、物理步长及每历史随机数保持不变。相同程序、反向顺序的独立种子确认得到 **1.946倍**连续吞吐；每20万粒子批次的 primary kernel 从22.320降至7.565 s，secondary从6.917降至3.498 s。所比较的输运整数、EM、电子计数完全一致，剂量相对 L1 最大5.01e-8。该场景队列峰值3,798,118 / 5,807,497个任务。这是调度一致性验收，不代表 carbon minibeam 剂量已验收。

增加共用 worker 数量、缩小错误诊断状态、C12主输运特化及缩小 secondary tail 的候选均未获得明确有效的收益，没有合入。实际 profiler 仍显示大量 local-memory 请求，每条已发射 warp 指令约11个活跃线程；仅降低静态寄存器数未提高运行速度。详见[完整报告](docs/ct_throughput_20261005.md)及[原始数值汇总](docs/ct_throughput_20261005.json)。

本轮 EM 搜索优化完成68个 broadbeam 完整批次，另有12个 CT minibeam 批次和其余六预设各两批回归。离子、EM、电子计数均保持一致，broadbeam 剂量相对 L1 最大4.823e-8，其余预设 smoke 最大1.081e-7。质量、能量、完成数、覆盖及零溢出检查通过。小粒子 smoke 不是 water gamma 或稳定吞吐验证。Water birth 提取每种粒子使用64个原生历史，仍需独立剂量/统计验证。先前清理引入的分支错误及较大统计量修复验证见[清理修正记录](docs/repository_cleanup_20261005.md#larger-statistics-correction)。

计时应分开冷启动、首批及常驻批次，报告 primary/s、kernel 时间、任务量和峰值内存，并记录 GPU、编译器、束流、种子及粒子数。比较相同入射 primary 口径，不能用电子任务或穿过准直器的 primary 替换。`scripts/benchmark_ct_devices.py` 已为四个 CT 预设统一使用原生连续批次，逐批检查，支持 `--smoke-only`；命令见 CT 报告。

每个候选前先查 [failed.md](failed.md)，只有满足记录中的新证据条件才能重试。FP32 原子累加顺序差异本身不否决调度改动，但输运/EM、能量、完成数和队列溢出审计仍必须通过。保留 TOPAS 参考和历史实验记录。Water 专项吞吐与剂量验收仍按要求推迟至 CT 验证之后。
