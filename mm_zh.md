# 材料与方法

修订日期：2026-09-14。描述当前工作树及指定配置，不视为冻结发布、临床准入或通用 Geant4 等价声明。[English](mm.md)。
历史数值只绑定对应的 executable、数据和配置，不能作为所有当前开关的验证。

## 1. 计算框架与当前配置

MAIGO 是 C++20 / SYCL 带电离子凝聚历史 Monte Carlo，在本地 RTX 2080 Ti
（`sm_75`）运行。GPU 查询 TOPAS/Geant4 派生表并重放核事件，不在线执行核内级联。
患者实现为 [transport_sycl.cpp](src/transport_sycl.cpp)，串行后端只支持子集。

参考采用 TOPAS 4.2.p3 / Geant4 11.3.2，含 standard_opt4 EM、QGSP_BIC_HP、
ion-INCLXX、elastic、stopping 和衰变模块。模块名不等于每个 projectile 实际挂载的模型；
已观察到 C12 使用 INCLXX。全离子弹性比较还需第 5 节所述的匹配扩展。

当前统一 EM 生产入口为[水](config/unified_water_production.yaml)和
[RT07575](config/rt07575_unified_em_production.yaml)：

| 设置 | 两个生产配置的显式取值 / 行为 |
|---|---|
| `em_model` | `g4_material_joint_v1`，水 / Schneider 共用一个包 |
| `enable_secondary_unified_em` | `true`：全部 18 种已支持带电离子使用统一 EM |
| 原发 / 次级涨落 | 均开启，`straggling_scale: 1.0` |
| `ct_secondary_exact_faces` | `true` |
| `secondary_species_grouping` | `true`；设为 `false` 恢复原调度 |
| `secondary_step_chunking` | `true`：64 次完整循环后压紧存活队列 |
| 非弹性 / 次级输运 | 开启；次级非弹性代数上限为 2 |
| 独立全离子核弹性 | 这两个生产配置不载入该库；属于研究选项 |
| 电子剂量 | 抽样 δ 能量局部沉积，不运行电子能量包 tracking |

其中 `primary_em_model: legacy` **不代表此时仍使用旧 stopping**：
`em_model: g4_material_joint_v1` 选择统一路径，原 primary-only 选项不是叠加模型。
反过来，只写 `em_model` 也不会自动开启次级统一 EM；配置结构中的次级开关默认 false，
完整次级性能实验均显式设为 true。更新 executable 不会自动升级所有历史 / 研究 YAML。
GPU 上 Legacy EM（`em_model: legacy`）已被拒绝；serial/cpu 后端（含 `transport_cpu`
与 CPU 测试）保留 legacy 路由。Joint water EM（`g4_joint_water_v1`）实现已删除，
其 YAML 键会被拒绝。

统一 EM 和种类分组已获准接入；低密度 production-cut 区、患者 Gamma，以及分组引起的
剂量差异仍待调查。执行质量接受与物理精度验收是不同结论。

## 2. 材料、源与几何

CT HU 映射到实际体素密度及 25 个 Schneider 组分区之一，使用 13 种元素组成。
CT 不走四分类回退。水采用独立、固定 SHA 的 G4_WATER，不冒充某个 Schneider 分区。
统一 EM 包覆盖水、25 分区及 18 种带电离子；密度节点查询保留材料 stopping 和阈值特性。

源由 TPS spot CSV histories、能散、发射度、扫描磁铁和患者摆位定义。分片前应与参考匹配
每个 spot 的整数 histories。TOPAS passive RotZ 使用 `R(+RotZ) × (world − Trans)`，
方向只旋转不平移。CT 打包、源摆位、剂量映射分别核对，不采用剂量拟合配准。
见[源与几何](docs/planning.md)。

## 3. 电磁过程

### 3.1. 当前统一模型的一步

1. 读取物种、动能、材料和密度，从统一包准备受限 stopping、range、离子修正、涨落参数和 δ 候选率。
2. 由原生 EM StepFunction、几何、核候选和抽样 δ 候选距离取最短允许步长，并应用低能停止 / 终止规则。
   统一路径中的原生 EM 步长取代旧 maximum-step 和相对能损限制。
3. 用 stopping 或 range 反演计算平均受限能损，加入离子修正，再抽样受限能损涨落。
4. 按实际步长消耗 δ 时钟。在候选点用能损后能量进行反应率接受检验，再抽电子能谱和形状因子 veto。
5. 只扣除一次连续能损与接受的 δ 能量，局部记分、推进并应用配置的 MCS，然后处理核候选并排入产物。

这是物理依赖顺序；kernel 内几何、记分和各过程分支交织执行。
随机流绑定 history 与过程身份；分组不会替换 Philox 或重启计数器。

### 3.2. 受限平均能损与密度

令 `T` 为粒子总动能 MeV，`E=T/A` 为 MeV/u，`h` 为步长 mm。
统一包保留原生受限 stopping、range 和 inverse-range 的 spline 段。
每个选中的密度节点先按实际密度 / 节点密度计算，再在相邻密度节点之间插值准备量和平均能损；
不能把当前模型概括成水 stopping 乘密度。

原生 StepFunction 参数为 `f`、`r_final`，剩余 range 为 `R`：

```text
h_EM = f R + r_final (1 − f) (2 − r_final/R)，R > r_final
h_EM = R，                                 其他情况
```

参数由每种离子的包记录提供；原发 C12 数据使用 `f=0.1`、`r_final=0.001 mm`，
不能未经核对套到全部物种。线性能损阈值同样来自包。

估计受限能损较小时用 `S_restricted(T) × h`；较大时用 inverse range 求末端能量，
反演期间保持步首质量 / 电荷缩放。离子修正在中间能量处查询，并含低能替换分支。
这不是旧的总 stopping 预测中点算法。结果受可用动能限制，包内最低动能可触发最终停止。

旧 `ct_primary_midpoint_stopping` 与材料特异次级 stopping 仍供非统一路径使用，表中保存：

```text
S1(section,ion,E) = S_extracted / [rho_reference/(1 g/cm³)]
S(T,section,rho) = S1(section,ion,T/A) × rho/(1 g/cm³)
```

这里密度没有重复计算。但这些旧表 / 开关不决定当前统一模型的受限平均能损。
未注册的额外重反冲仍用专门的 stopping 路径。见[统一查询与能损](include/carbon/unified_em_view.hpp)。

### 3.3. 涨落与显式 δ 候选

受限连续能损按适用情形使用包内 IonFluc 或 Universal/Urban 相关采样器。
原发及已支持次级均开启原生涨落，scale 为 1.0；旧 CT 的 scale 1.2 和次级 CSDA
属于历史配置，不是这两个生产入口。

δ 时钟抽样 `tau = −ln(U)`，沿实际路径消耗候选光学深度；局部候选距离为
`tau_remaining / Sigma_proposal`。候选不一定产生电子，还要经过能损后的率接受检验及
能谱 / 形状因子接受步骤。材料或密度变化使缓存率失效，但保留未消耗光学深度。
平均自由程是统计尺度，不是固定碰撞位置。

```text
DeltaT = 抽样受限连续能损 + 接受的 δ 电子动能
T_out  = T_in − DeltaT
```

受限 stopping 排除了显式处理的阈值以上转移；δ 阈值随材料 / 密度变化，不能将水中约
57 keV 的示例用于所有 CT 分区。当前 δ 能量局部沉积，不再叠加旧比例电子响应。
`em_macro_ticks` 已不是当前支持的开关；撤回的宏 tick / RNG 实验不代表生产算法。

### 3.4. 库仑多重散射

配置的 Highland 近似使用分区辐射长度和局部密度：

```text
t = rho × (h/10) / X0_mass
C = max(0, 1 + 0.038 ln(t Z²/beta²))
theta0 = 13.6 MeV × Z/(beta p c) × sqrt(t) × C
```

在局部横向坐标系抽取偏转，再旋转到粒子方向；空间 / 方向更新遵循选中的输运分支。
散射 scale 默认 1.0，两个生产配置没有调参。此近似不等于 Geant4 完整 msc，
也不能替代强相互作用核弹性。见[MCS](include/carbon/multiple_scattering.hpp)。

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
缺通道不以近邻靶替代，不整体缩放产物动能。如果 EM 能损后没有有效靶，
则作为 null candidate 保留剩余动能继续，不重放事件、不局部倾倒能量。
冻结 generation 设置为 2；He6/B8/C10 遵循声明的 EM-only 核策略。

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
| 库仑散射 | 凝聚 Highland 模型 | 配置的 Geant4 电磁过程 |

**旧 topas10x 不能直接作为新模型的匹配参考。** 审计确认旧列表只有 p、d、t、He3、alpha 有核弹性，GenericIon/C12 未挂载；新增 `CarbonIonElasticPhysics` 后补齐。匹配重跑保留原电磁、非弹性、stopping、衰变模块以及 CT、源、3D 剂量网格和总 histories，仅省略 LET scorer。

基准库为137能量节点、每节点512末态。运动学、host/device查询、水闭合及RT07575的6481909 histories分片验证通过且零溢出；2048样本及加密网格候选也通过分片闭合。低能截面起始区插值、额外反冲贡献及全统计匹配剂量验收仍未完成，尚未切换默认生产配置。

使用 [CT研究配置](config/rt07575_elastic_research.yaml) 或 [水研究配置](config/unified_water_elastic_research.yaml)，不是旧 `enable_nuclear_elastic` 开关。11.3.2 Release不含新弹性数据。详见[实现与数据](docs/all_ion_elastic.md)、[验收记录](evidence/step-31/elastic-production-validation-20260911/README.md)和[9月12日参考迁移](evidence/step-31/elastic-migration-20260912/README.md)。

## 6. 电子与中性产物

当前统一 EM 抽样 δ 转移，但不空间跟踪这些电子。受限能损与 δ 能量形成局部剂量；
局部沉积本身是近似，在界面、横向尾部和 minibeam valley 尤其需要验证。
当前没有可再通过关闭而大幅提速的完整电子 tracking kernel。

仓库另有旧 section-0 delta-tail 搬运、材料电子家族 / 能量包重放候选。
能量包沿记录状态和续接路径搬运已经预算的能量，不让离子再损失第二份能量。
这些候选不在统一生产入口中启用，也不能直接叠加到受限加显式 δ 模型。
材料响应精度仍未验收。见[能量包输运](include/carbon/electron_packet_transport.hpp)和[电子计划](plan2/README.md)。

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

当前 CT Gamma 口径：

- 根据 RTSTRUCT 构造 BODY mask，只评价 BODY 内参考体素中心；记录 ROI、轮廓栅格化 / 插值方法及 mask SHA。
- 按现用 10% 剂量阈值，评价 `BODY ∩ {Dref >= 0.1 Dmax}`，`Dmax` 为参考全体积最大值。
  BODY 归属和剂量阈值是两个独立筛选条件。
- Global 容差以 `Dmax` 为基准，local 以查询点参考剂量为基准；粒子数和几何匹配后评价
  3%/3 mm、2%/2 mm、1%/1 mm、3%/0 mm。
- DTA > 0 时搜索步长为 `DTA/10`，分别 0.3、0.2、0.1 mm，使用三线性插值。
  BODY 限定参考查询点，不把 GPU 的 BODY 外剂量强制置零。格点搜索不等于解析连续最小值。
- DTA = 0 时只比较同体素，不受搜索步长影响。

不拟合剂量归一或配准。9 月 5 日无 BODY、固定 0.5 mm 的结果保留为历史证据；
部分旧评价脚本仍使用该口径，不能不改参数直接作为当前方法。
可参考[BODY 评价实现](benchmark/benchmark20260912/RT06423_replica01/evaluate.py)。

质量检查包括有限数值、数据来源、抽样 audit、能量记账和零 queue overflow。
overflow 使该片无效，必须拆分重跑。全局能量闭合不证明空间剂量正确，也不等于每个核顶点 Q 值闭合。

## 8. 种类分组与实测吞吐

### 默认次级续跑（2026-09-14）

两个生产预设均启用 `secondary_step_chunking: true`，与种类分组一起使用。
每次 kernel 最多执行 64 次完整次级循环，保存存活轨迹并稳定压紧其索引；
队列少于 8192 时直接跑完剩余轨迹，不缩短或合并物理步。
续跑保留能量、位置/方向、RNG 计数、δ 时钟、材料缓存、待写回沉积及诊断累计量；
只有真正终止才做末端计分，每代完成后才开始子代。设为 `false` 恢复整条轨迹
单次运行且不分配续跑缓冲；种类分组由自己的开关控制。

默认构建每条当前代次级需要 336 字节状态及约 20 字节索引/标志。
实测 288 万次级约增加 1.03 GB 显存。分配失败会停止运行，应减少每个 shard 的
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
（+37.0%），最大差为峰值的 0.0000966%。已重建的日常生产二进制也通过含弹性
1M 检查，吞吐为 28,534 histories/s。
参见[验证记录](benchmark/runtime_breakdown_20260914/SEGMENT_VALIDATION.md)及
[正式接入](benchmark/runtime_breakdown_20260914/SEGMENT_PRODUCTION.md)。



`secondary_species_grouping: true` 在 GPU 上对每代次级建立索引排列，分为 18 种离子
及其他产物共 19 桶。Histogram、prefix sum、scatter 保留粒子记录、parent history 和
RNG stream，子代仍在下一代处理。此前的物种专用 kernel 分裂候选仍未接入，额外索引内存约为每队列槽 4 字节。

两个生产 YAML 显式开启；配置结构默认 false，以兼容旧配置。日志输出模式及分组时间；
授权接入后质量报告仍保留 `secondary_species_grouping_accuracy_pending`。

RT07575，100 万原发，完整次级统一 EM **加研究模式核弹性**，两轮均值：

| 调度 | histories/s | 相对提升 |
|---|---:|---:|
| 原调度 | 11,154 | — |
| 种类分组 | 14,691 | 31.7% |
| 分组 + 隔离的 kernel 分裂 | 14,828 | 32.9%；分裂未接入 |

计时分母已包含分组开销。这不是两个不启用独立弹性的生产 preset 的测量值。
分组与基线的最大体素差为峰值的 0.0203%，原调度自身重复差为 0.0047%。
计数和 EM audit 相同，但较大差异原因仍待追查；授权接入不等于剂量等价性门槛通过。

正式入口另验证 RT07575 200k、水 50k，分组开 / 关各一次，完整次级 EM、不含独立弹性。
四次均通过执行质量、零 overflow，步数 / audit / 反应计数一致；最大剂量差分别为峰值的
0.0000133% 和 0.000172%。它们不能消除上述含弹性 1M 的未决差异。
见[实验与接入记录](benchmark/benchmark20260914/secondary_schedule/README.md)。

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

本次只更新方法和研究建议，没有新增输运近似或物理精度验证；已有生产接入检查列于第 8 节。

## 参考

- [FRED carbon 文献分析](docs/FRED_Carbon_Fragmentation_Model.md)
- [统一 EM 数据与模型](docs/physics/unified_em_v1.md)
- [旧 primary-water 联合模型，已从代码移除](docs/physics/water_joint_em_v1.md)
- [历史物理规格](docs/TOPAS_GPU_Physics_Model.md)
- [记分契约](docs/scoring_validation.md)与[旧稿归档](docs/archive/README.md)
