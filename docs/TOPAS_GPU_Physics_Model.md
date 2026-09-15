# TOPAS / Geant4 与 GPU 物理模型

更新日期：2026-09-05。本文是当前实现说明；不是整个 Geant4 physics list 的复刻声明。
数据最低版本由 [AGENTS](../AGENTS.md)、[bundle](../data/schneider/schneider_physics_bundle_v2_1.json)
和 [manifest](../data/schneider/v2_1_data_manifest.json) 控制，结果以 [冻结索引](results.md) 为准。
旧 physics/methods 原稿保存在 [归档](archive/README.md)。

## 1. 参考模型与研究范围

参考 TOPAS 4.2.p3 / Geant4 11.3.2 的模块为：

```text
g4em-standard_opt4
g4h-phy_QGSP_BIC_HP
g4decay
g4ion-inclxx
g4h-elastic_HP
g4stopping
g4radioactivedecay
```

模块列表不是各粒子实际调用模型的清单。此前 runtime 普查中参考 C12 调用 INCLXX，
p 可调用 BIC；不能因 list 含 BIC 就把剩余差异称作“C12 INCLXX vs BIC”。
GPU 重放提取的相关末态，不运行 INCLXX/BIC。相同模型来源也不保证步进、
能量节点抽样、次级输运和记分完全等价；必须分层比较。

本次三病例验证为 C12 CT research dose，非全部离子、LET、neutral 或任意几何准入。
`validation_scope.production_generalization=false` 不能因 Gamma 高而撤销。

## 2. 材料与数据路由

| 过程 | 当前 Schneider CT | 保留 water 路径 |
|---|---|---|
| 材料 | HU → 连续密度 + 25 section，元素组成用全局 LUT | 水/指定 phantom 材料 |
| C12 stopping | SCHNSTOP v1 的 section mass stopping × 局部密度 | water primary stopping |
| 次级 stopping | water-ion stopping × CT 材料因子 × 局部密度 | water-ion stopping |
| MCS | primary/secondary 共用 25-section X0 选择，使用局部密度 | water X0；Highland MCS（旧2GR已退役） |
| 非弹性率与靶选择 | SCHNRATE/SCHN2RAT v3，section/projectile/target partials | water XS 与对应 CINEL02 rates |
| 核末态 | CINEL03 接口读取 CINPKG04 v4，精确靶通道 | CINEL02 water 事件包 |
| 涨落 | 配置选择的凝聚能损采样 | 可选已保留的 TOPAS water fluctuation 数据 |
| section-0 C12 delta-tail | 窄范围剂量重分配，见第 6 节 | 不适用；不是通用水电子包 |

CT 不得退回四分类或 water 核数据。代码仍包含历史分支不代表允许在当前验证中使用。
启动层仍加载 water primary stopping/XS；次级初始化仍依赖
`data/ion_stopping_power_water_geant4_11_3_2.csv`。这些是真实依赖，
不能因为 CT H/O 靶通道存在就删除。精确文件及外部 water 包见
[ACTIVE_DATA](../data/ACTIVE_DATA.md)。

Schneider 源文件定义 25 sections 和 H/C/N/O/Mg/P/S/Cl/Ar/Ca/Na/K/Ti。
同 section 组成固定、密度随 HU 变化。不能用材料探针的代表密度作为该 section
所有 CT 体素的密度，也不能用它反算患者 TOPAS Edep。
独立材料验证应核对边界、组成和连续密度，不仅核对 section ID。

## 3. 带电输运、stopping 与 MCS

当前 GPU 入口是 [transport_sycl.cpp](../src/transport_sycl.cpp)。
步长受配置最大长度、相对能损、材料/体素边界、核光学深度与终止条件共同约束。
三病例冻结配置的 `maximum_step_mm=0.5`、
`maximum_relative_energy_loss=0.005`；不是旧 methods 的 0.1 / 0.001。

Schneider C12 表提供按分区的质量阻止本领与 range 数据，运行时按实际密度求能损。
它不是 14 种抛射体 × 25 分区的完整 isotope stopping 表。
次级使用水中 ion 表及材料因子，并在慢化步中计算相应 stopping；
不能声称逐 isotope 的 CT dE/dx 全部直接来自精确 Geant4 材料提取。
源到 CT 的 upstream air 能损独立计算，保留空气 stopping 输入。

Highland MCS 使用局部面密度和 X0。当前 CT primary 与 secondary 共用
[select_transport_radiation_length_g_per_cm2](../include/carbon/multiple_scattering.hpp)
选择 25-section LUT；三病例 `enable_ct_material_mcs=true`。
四分类和 water X0 不是当前 CT 模型。Highland 仍是近似，不是 Geant4 msc 算法复刻。

能损涨落由 [straggling.hpp](../include/carbon/straggling.hpp)、
[energy_loss_fluctuation.cpp](../src/energy_loss_fluctuation.cpp) 及运行开关共同确定；
不要把全部可选 sampler 统称为 Vavilov 或 G4IonFluctuations。
三病例 primary straggling 开启，冻结 `straggling_scale=1.2`；
secondary straggling 默认关闭。记录现有配置不等于授权调 scale。
新运行必须冻结实际解析配置，而不是只保存有默认遗漏的 YAML。

## 4. 材料相关核反应与相关末态

最低 CT 栈：

- primary rate：`schneider_inelastic_rates_v2_1.bin`，SCHNRATE v3；
- primary events：`cinel03_c12_targets_v2_1.bin`，CINPKG04 v4；
- secondary rate：`secondary_inelastic_rates_v2_1.bin`，SCHN2RAT v3；
- secondary events：`cinel03_secondary_targets_v2_1_14p.bin`，CINPKG04 v4；
- stopping：`schneider_stopping_v1.bin`，SCHNSTOP v1。

核反应率按 projectile × section × energy × target 的 partials 处理。
域外 partial 在查询时屏蔽；总 hazard 为有效 partials 之和乘密度，密度只进入一次。
光学深度跨步推进；每个 target 的选择概率由对应 partial/total 决定。
边界 mask 与有限步长处理仍属实现近似，不能把域屏蔽描述成独立物理证明 XS 为零。

[CINEL03 lookup](../include/carbon/inelastic_package_v3.hpp) 的 key 为
`(projectile Z, projectile A, target Z)`：

1. 缺 projectile/target、空通道、能区外均有显式状态，不 alias 到 O。
2. 精确节点命中；域内相邻节点以 `P(E1)=(Eq−E0)/(E1−E0)` 随机选择。
3. 非精确节点跨越 >5 MeV/u 的 gap 拒绝；不做无界 nearest-neighbor/clamp。
4. 在所选节点抽取完整相关事件，并把局部产物方向旋转到入射方向。
5. 不按查询能量整体缩放事件 KE，也不以 Q/质量亏损强行补成局部剂量。

candidate hazard 与 post-EM 能量并不总相同。post-EM 无有效靶时属于
`PrimaryPostEmNullCollisions` / `SecondaryPostEmNullCollisions`：
无 lookup、无局部倾倒，保留粒子剩余能量继续；它不是有效 replay 或 stopped。
核查应使用 candidate 的完整互斥分类，不只报 replay hit rate。

v2.1 的 residual-NEED/floor 例外有冻结记录；“query miss=0”仅证明当前
masked 域内运行闭合，不证明 0–430 MeV/u 任意能量的全部物理反应都覆盖。

## 5. 次级、代数、scope 与不覆盖过程

带电碎片入 GPU 队列并继续 EM 输运；三病例 generation 配置为
`cinel02_max_secondary_inelastic_generations=2`（此历史键名仍控制 CT 路径）。
generation 的 hazard/child 入队条件见实现，不能把数值 2 理解成任意深度 cascade。
born 应严格等于 queued + cutoff + overflow 等互斥 terminals；Be6 特例单列。
overflow 不是可接受的物理终止，必须拆分重跑。

bundle registry 有 14 个 projectile。He6/B8/C10 的声明策略为
`secondary_out_of_scope_nuclear_policy: em_only`：
继续带电 EM 输运、禁用 scope 外次级核反应，不把整份 KE 当场倾倒。
其他不支持产物、neutral、Q 与截断诊断须按 ledger 和 scope 分开报告；
信息性分项不能再加到已包含它们的 legacy untracked 总项。

三病例未启用独立 nuclear elastic；非弹性事件有角分布不代表覆盖了弹性散射。
通用 e±/gamma/neutron 输运与衰变链也未由这批 dose 结果验证。
全局账本闭合是程序验收条件，不等于每个反应顶点都满足精确的质量/Q 守恒。

## 6. 电子剂量响应：已用版本与未验证候选

当前严格剂量栈保留 `schneider_section0_c12_delta_tail_v1.csv`：
TOPAS/Geant4 派生、仅 primary C12 / Schneider section 0，
提取能点 150/200/225 MeV/u。它从已经记入 primary 能损的能量中搬运一部分，
不是额外创造电子能量，也不是完整电子输运；域外 lookup 返回零搬运。
源码对 source eligibility、目的材料与 scorer 逃逸有约束，不能推广到其他材料/碎片。

2026-09-05 三病例冻结 executable 包含 entrance-mask candidate；
原有横向响应表没有升级。当前工作树另有纵向响应代码/表候选，
**未被该三病例结果验证或提升为最低数据栈**。
联合纵向/横向响应及跨密度、出生条件、几何收敛门禁由
[plan2](../plan2/README.md) 控制；该计划尚未完成。
不得把当前 Gamma 残差直接认定为“全由电子造成”，也不得据此调 package 角谱。

## 7. 记分与验证

主输出为 3D DoseToMedium，转换使用每体素质量；IDD 由 3D 横向求和。
能量 ledger、in/out-grid split、origin dose、lookup/overflow 计数与 Gamma
互补，任何单项不替代其余验收。详见 [scoring_validation](scoring_validation.md)。

运行前严格 verifier，运行后逐 shard accepted/overflow/provenance 检查；
数据升级还需 host/device、50k、配对单 shard 与零溢出完整验证。
旧 `physics_profile: best/medium/fast` 当前被配置拒绝，不再作为推荐入口。
当前数值只在 [results](results.md) 维护，不继承旧 98% 单 shard 的模型归因结论。
