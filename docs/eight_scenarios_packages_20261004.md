# 八个默认场景的运行路径、数据包与退出清单

核对日期：2026-10-04。依据当前8个默认YAML、config展开规则、运行加载器、实际文件头与包metadata；初始清单未运行蒙卡，后续broadbeam默认切换的运行验证见文末报告。本清单排除第9个 `carbon_soft_minibeam_independent_trial` 研究配置，但在迁移说明中列明其用途。数据路径均相对仓库根目录；同目录不代表目录内每个历史文件都被使用。SHA、metadata、channels、索引指向的payload属于包的一部分。

## 1. 当前八条路径

| 场景 | 配置入口 | 当前处理路径 | 电子及束线等场景包（另加下节基础包） |
|---|---|---|---|
| Proton water broadbeam | config/proton_water_broadbeam.yaml | broadbeam_fast；FE；水与核反应/带电次级；低密度tracking开启，均匀水选区为空，电子局部沉积 | 无独立空间电子包；water_unified材料描述 |
| Carbon water broadbeam | config/carbon_water_broadbeam.yaml | broadbeam_fast；Gaussian FE；水与核反应/带电次级；低密度tracking开启，均匀水选区为空，电子局部沉积 | 无独立空间电子包；water_unified材料描述 |
| Proton CT broadbeam | config/proton_ct_plan_fe.yaml | broadbeam_fast；分段Gaussian FE；RT07575 TPS输入；低密度＋5mm共用FP32电子输运 | proton_independent_v1出生＋shared_independent_v1；上游空气proton_air_stopping |
| Carbon CT broadbeam | config/carbon_ct_plan_fe_joint.yaml | broadbeam_fast；分段Gaussian FE；RT07575；低密度＋5mm共用FP32电子输运 | carbon_independent_v2出生＋shared_independent_v1；上游空气表；不再加载旧carbon_joint/tail |
| Proton water minibeam | config/proton_water_minibeam.yaml | compact铜/空气→水；global Urban；water_response_v2；含实验中子输运 | proton_copper_20260926/；其urban_water_copper.bin；air_urban_v2_20260926/；proton_air_ionization_20260927/；water_electron_response_20260926/proton/；neutron/native＋neutron/recoil |
| Carbon water minibeam | config/carbon_water_minibeam.yaml | combined默认主调度；FE公共选择＋显式水primary/secondary C12 Urban；water_response_v2 | carbon_copper/＋carbon_copper_elastic_20260926/；carbon_urban/active_copper005.csv、active_water005.csv；air_urban_v2_20260926/；water_electron_response_20260926/carbon/ |
| Proton CT minibeam | config/proton_ct_minibeam.yaml | compact铜/空气→CT；Urban密度缩放；FP32独立birth/joint/charged/photon；group32 | proton_copper_20260926/；proton_air_ionization_20260927/；ct_urban/ct_copper_52.bin；air_urban_v2_20260926/；ct_electron/proton_independent_v1/＋shared_independent_v1/ |
| Carbon CT minibeam | config/carbon_ct_minibeam.yaml | compact铜/空气→CT；Urban密度缩放；C12 compact v7材料电子响应；group32 | carbon_copper/＋carbon_copper_elastic_20260926/；carbon_urban/active_copper005.csv；ct_urban/ct_copper_52.bin；air_urban_v2_20260926/；ct_electron/carbon/index.json及源身份文件＋carbon_compact/ |

`minibeam_water_delta_response_model: local`在CT独立/材料电子预设中不是“所有电子局部沉积”的证据：实际由material/independent电子选择接管。应按生效分支读配置。

统一入口为 `scripts/run.sh PRESET` → `src/main.cpp`；原生重复批次由 `src/batch_run.cpp` 组织，输运主体为 `src/transport_sycl.cpp`。共享铜/空气主段包含 `src/detail/sycl_primary_beamline.inc`。独立FP32电子worker为 `src/detail/sycl_independent_electron_fp32.inc`；carbon compact通过 `MaterialElectronDeviceBank`/`MaterialPreparedReader` 加载并走材料响应worker；broadbeam joint队列为 `src/detail/sycl_fe_joint_replay_kernel.inc`。这些入口不是八套独立引擎。

### Broadbeam共用关系澄清

四个broadbeam均走 `transport_profile: broadbeam_fast` 和Gaussian FE，同一 `broadbeam_electron_tracking` 开关默认true。用户确认沿用密度≤0.1 g/cm³＋5mm邻域；均匀water没有选中区域，保持局部沉积且不加载电子包。CT使用 `transport_sycl_impl<true,true,3>` 及共用FP32电子worker，默认不再用旧joint/tail。设false时使用无电子包的局部沉积分支；water/CT主输运、核反应、带电核次级继续共用既有框架。

只接管主proton/C12的电离电子，带电核次级的电子损失仍局部沉积。新CT路径按RT07575 local1%/1mm >99%验收，proton99.9120%、carbon99.8677%；速度低于局部沉积。全局局部沉积carbon为96.2842%，关闭开关不代表同样通过剂量标准。默认切换及开关验证见[报告](broadbeam_tracking_defaults_20261004.md)，之前joint消融结果作为历史证据保留。

## 2. 基础物理包：当前仍用，不能按旧包删除

八组共同使用：
- `data/em/unified_em_v1.bin` 与 `data/em/unified_em_delta_moments_v2.bin`。Carbon水minibeam没有显式列出后者，但配置根据EM路径补齐。版本号v1/v2分别属于不同组件，不能以较小编号为由删除。
- `data/schneider/secondary_inelastic_rates_v2_1.bin`、`cinel03_secondary_targets_v2_1_14p.bin`及channels metadata：共享带电核次级反应数据。
- `data/HUtoMaterialSchneider.txt`及各自physics bundle中的材料/通道身份信息。

四组proton使用：
- `data/proton_therapy_20260926/proton_rates.bin`、`proton_cinel03.bin`、`proton_cinel03.channels.json`、`physics_bundle.json`。
- 同目录 `proton_water_stopping.csv`：仍由main加载，作为独立核/反冲参考能量网格；水/CT主能损来自Unified EM，不能因此删除该CSV。
- 同目录 `all_ion_elastic_proton_v1.bin`、`elastic_recoil_stopping_water75.bin`：核弹性/反冲响应。

四组carbon使用：
- `data/schneider/schneider_inelastic_rates_v2_1.bin`、`cinel03_c12_targets_v2_1.bin`、其channels、`schneider_physics_bundle_v2_1.json`。
- `data/stopping_power_water_geant4_11_3_2.csv`：启动/参考网格。
- 两个carbon minibeam另外启用 `data/schneider/all_ion_elastic_v1.bin` 与 `elastic_recoil_stopping_v1.bin`；两个carbon broadbeam没有显式启用这两个弹性包。

四组water另用 `data/water_unified/g4_water_material.json`：核反应/组成几何参考。此文件名中的g4_water不代表将EM水参考改回78eV；项目EM仍是Water_75eV。

有核弹性反冲或中子输运的路径还用 `data/ion_stopping_power_water_geant4_11_3_2.csv`；minibeam铜次级需要对应ion铜停止本领与 `data/ion_cross_sections_copper_geant4_11_3_2.csv`。`use_particle_specific_stopping_power`开关被丢弃，不代表这些CSV数据本身没有其他活跃用途。

所有minibeam的铜包需保留所选primary stopping、inelastic rate、elastic、cascade、ion stopping等成员。Proton铜包还提供空气停止本领；carbon空气使用 `data/stopping_power_air_geant4_11_3_2.csv`。CT broadbeam的RT07575输入另外启用相应空气停止本领CSV。

## 3. 五组当前活跃的电子数据组合

| 组合 | 精确入口与附属内容 | 服务场景 |
|---|---|---|
| Proton独立电子 | `data/ct_electron/proton_independent_v1/birth_bank.bin`＋`shared_independent_v1/schneider_joint.bin`＋`shared_independent_v1/{charged,photon}/index.txt`及所有节点operator | Proton CT minibeam＋Proton CT broadbeam tracking |
| Carbon compact | `data/ct_electron/carbon/index.json`及源metadata/身份/文件大小依赖＋`data/ct_electron/carbon_compact/` v7 payload | Carbon CT minibeam默认 |
| Proton水响应 | `data/water_electron_response_20260926/proton/{table.bin,table.metadata.json,table.paths.bin}` | Proton water minibeam |
| Carbon水响应 | `data/water_electron_response_20260926/carbon/{table.bin,table.metadata.json,table.paths.bin}` | Carbon water minibeam |
| Carbon独立电子 | `data/ct_electron/carbon_independent_v2/birth_bank.bin`＋同一`shared_independent_v1/`传播包 | Carbon CT broadbeam tracking |

这是五组数据组合、三类电子处理方法（独立输运、compact、水响应回放）。Proton/Carbon水响应是两个粒子专属包，不能彼此替代。

`data/ct_electron/carbon_independent_v1/`不在八个默认预设中使用，但属于要保留的新研究包；由第9个carbon软组织试验配置加载carbon birth＋shared。它不是旧版兼容包。Carbon CT minibeam默认尚未切换；broadbeam默认使用补齐82节点的v2出生包，已完成RT07575 scoped验收。

`carbon_compact`当前是指向 `/home/wuwei/.cache/maigo/carbon_endpoint_inline_v7_20261003` 的有效部署链接，不是废弃路径别名。

## 4. 应退出运行发行目录的旧数据与兼容接口

以下针对“只交付当前八默认场景＋保留carbon独立新试验”的目标；历史复现数据可以单独归档，本轮未删除。

| 项目 | 当前判定 | 处理 |
|---|---|---|
| `data/ct_electron/proton/` | 旧proton材料响应；当前独立proton路径零legacy依赖，八默认场景均不引用它 | 从当前运行发行包移除；历史对照/来源记录单独归档 |
| `data/ct_electron/proton_packed/` | 旧proton压缩运行包；默认已由birth＋shared替代 | 同上，并退出PROTON_EM_2旧运行入口 |
| `proton_independent_v1/{charged,photon,schneider_joint.bin}`兼容符号链接 | 新配置已直接引用shared目录；仅为旧路径继续可用 | 更新旧脚本及check_electron_packages.py/兼容清单后删除链接；不会节省第二份大包容量 |
| `data/hu_stopping_power_lut_geant4_11_3_2.csv`的CT LUT兼容入口 | carbon CT broadbeam YAML仍有路径，但load_config直接丢弃该键，prefetch也跳过 | 删除旧键和当前发行依赖；历史表可归档 |
| `data/schneider/schneider_stopping_v1.bin`、`data/proton_therapy_20260926/proton_schneider_stopping_v1.bin` | nuclear bundle里的stopping_table是历史metadata，当前loader不作主能损输入，prefetch明确跳过 | 退出当前运行包；保留来源记录或同步整理bundle说明，不能删除同目录仍用的核反应包 |
| 旧material PREPARED_1、water_response_v1入口 | 八默认配置不使用；generic prepared不是当前carbon v7 | 删除对应历史运行选择及仅服务它的代码，不删除共享读取/审计代码 |
| CT v1/v2、material index schema1兼容读取 | 当前默认配置引用的CT都是v3，carbon材料index为schema2 | 先约束当前输入格式，再删除旧读取分支 |
| 无效旧EM键与同义配置键 | primary_em_model等占位、use_particle_specific_stopping_power、scorerLET/tps_gantry_angle_deg等 | 去掉无效键、统一公开拼写；相关数据文件是否活跃必须另判 |
| 历史实验开关/重复目录环境变量 | 包括旧prepared/packing开关、失败的精度/性能候选入口 | 分别退出；保留最新正式设置及必要诊断，不减掉质量/能量/队列检查 |

## 5. 需要先迁移，不能直接删的兼容负担

- **MAIBIR01**：当前proton最新birth数据仍是此格式，carbon新birth是MAIBIR02。先为proton增加Z/A标签、更新SHA且验证payload不变，再只留MAIBIR02读取器。
- **combined调度**：当前carbon water minibeam仍用。不能因proton已用compact就删除combined。
- **FP32的参考方程头文件**：当前生成器仍读取它们；取消precision=0独立运行入口不代表可以删除方程源、loader或原始磁盘结构。
- **carbon v7源依赖**：当前bank会读取index/metadata并对源state-reference文件取大小，派生身份也含原始SHA。不能仅保留compact二进制就删除整个`carbon/`目录。需要先设计自包含v7后继包并验证。
- **旧迁移脚本参与构建指纹**：`scripts/material_prepared_sources.txt`包含migrate_carbon_electron/bitmap/endpoint等；CMake对清单文件全部求SHA形成CARBON_MATERIAL_PREPARED_VERSION。直接删除会导致配置失败；直接改清单也会改变现有carbon包的查找身份。应同步明确派生格式/算法身份并重建或受审计迁移，而非关闭完整性检查。
- **两个水响应包、carbon compact**：仍服务默认minibeam，需要迁移后才能退出。`carbon_joint`和section0 C12 delta-tail已退出四个broadbeam默认路径，本轮未删除，保留历史验收复现。

## 6. 场景输入（与物理包分开）

| 场景 | CT/束流输入 |
|---|---|
| 两个water broadbeam、proton water minibeam | YAML中的均匀水几何/源参数 |
| Carbon water minibeam | `inputs/carbon_water_minibeam/spots.csv`＋YAML束流光学 |
| Proton CT broadbeam | `inputs/proton_rt07575/patient_packed.bin`、`beam_model.csv`、`part_01/spots.csv`；不是full plan默认入口 |
| Carbon CT broadbeam | `inputs/carbon_rt07575/rt07575_packed.bin`、`beam_model.csv`、`spots.csv` |
| Proton/Carbon CT minibeam默认 | `inputs/lung_ct/ct.bin`＋YAML平面源；RT07575 full plan由其独立benchmark配置提供 |

本轮新增清单，不改变这些默认场景。详细显式路径键附录：`out/eight_scenarios_packages_20261004/declared_inputs.json`；该JSON把声明路径和忽略/非当前dose用途标记分开，不声称仅扫描YAML就穷尽所有传递依赖。

2026-10-04后续：按用户明确要求，四个broadbeam默认统一tracking开关；两个CT采用已验收共用电子路径，两个water选区为空。原independent_trial配置保留为参考入口，与CT默认电子设置相同。[接入与验收报告](broadbeam_shared_electrons_20261004.md)。
