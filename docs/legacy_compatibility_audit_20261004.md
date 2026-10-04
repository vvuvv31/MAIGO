# 当前运行路径与旧版兼容项清单（2026-10-04）

本轮按用户“先看看”的要求做静态核对：读取当前源码、9个配置（8个场景预设＋1个carbon软组织研究预设）、实际数据文件头和默认构建选项；没有修改运行代码、预设、数据包或可执行文件，没有启动蒙卡。完整配置快照与源码SHA见 `out/legacy_compatibility_audit_20261004/inventory.json`。

结论：可以明显缩减兼容层，但“只保留最新内容”应按每个场景当前有效实现划分。当前八组预设尚未全部迁移到同一个电子运行模型；删除所有带legacy/reference/diagnostic名称的内容会破坏现有运行。

## 当前真正使用的路径

| 场景 | 当前电子处理 | 主调度/备注 |
|---|---|---|
| Proton CT minibeam | 独立birth＋shared joint/charged/photon，precision=4 | compact、group32；不依赖旧material replay库 |
| Carbon CT minibeam（默认） | C12 compact v7材料电子响应，tiered_small_decoded | compact、group32；仍是活跃依赖 |
| Carbon soft minibeam trial | C12独立birth＋同一shared包，precision=4 | compact、group32；软组织研究入口，不能直接替代完整头部病例 |
| Proton water minibeam | proton专用water_response_v2 | compact；尚未迁移到shared独立传播模型 |
| Carbon water minibeam | carbon专用water_response_v2 | 未显式指定pipeline，默认combined；不是当前proton的compact/Urban组合 |
| Proton water/CT broadbeam、carbon water broadbeam | broadbeam_fast；电子损失按各预设现有局部处理 | FE路径；不是旧CT minibeam电子库 |
| Carbon CT broadbeam | 已有C12 joint response，queued | RT07575剂量验收已通过；通用研究质量标记仍在，不能因为文件名含diagnostic而删除 |

## 第一批：可收敛的纯兼容和历史入口

| 项目 | 当前证据 | 建议 |
|---|---|---|
| 被读取后丢弃的旧EM键 | `src/config.cpp:2410` 起：primary_em_model、enable_secondary_unified_em、ct_use_density_mass_spr、ct_stopping_power_scale、旧材料停止本领表路径等；没有旧EM运行后端 | 从预设去除残留键，再删除兼容接受逻辑，让未知旧键明确报错 |
| 预设里的无效残留 | carbon_ct_plan_fe_joint仍有ct_hu_stopping_power_lut_file；4个预设含use_particle_specific_stopping_power。上述键被解析器丢弃 | 优先删除配置占位，不能将这些路径继续当作活跃物理输入 |
| 历史YAML别名 | ct_schneider_c12_cinel03_file、enable_minibeam_primary_c12_roi_scoring、ct_secondary_exact_faces_diagnostic、tps_gantry_angle_deg、scorerLET等 | 统一为现有正式键；carbon水预设当前仍使用后两项，需先改配置 |
| 同义入口 | tpsSource/tps_source当前均接受，且现有计划预设使用tpsSource | 保留一个公开拼写即可；不能误以为C++字段enable_tps_source已是可用YAML键 |
| 已退役铜散射选择 | minibeam_copper_mcs_model=urban明确仅供历史复现；当前minibeam预设使用urban_v2 | 移除旧urban选择和只服务该选择的实现，保留urban_v2与实际物理审计 |
| CT旧二进制格式 | loader支持CCTG v1/v2/v3；本轮9配置所有引用CT的文件头均为v3 | 活跃预设可统一v3；v1/v2读取和四类材料兼容可在检查其他入口后移除 |
| 旧material index schema | loader支持schema1/2；默认carbon CT index实测schema2 | 只保留schema2可作为候选；不要删除carbon v7实际使用的index/读取器 |
| 旧proton packed与generic prepared | MaterialPreparedReader支持PREPARED_1、PROTON_EM_2及CARBON_EM_7；当前proton独立路径不使用前两种 | 拆除旧proton运行分支和generic历史加载分支，保留carbon v7必需部分 |
| 旧环境变量注入 | CARBON_MATERIAL_PREPARED_DIR、CARBON_ELECTRON_CARBON_COMPACT_DIR、CARBON_ELECTRON_PREVALIDATED、CARBON_ELECTRON_PACK_EM_ONLY | 当前目录选择已有YAML入口；可去掉环境兼容，保留明确配置、完整SHA及谱系校验 |
| 旧版制包迁移工具 | migrate_carbon_electron/bitmap/endpoint；当前运行只接受carbon v7，不是同时运行v4/v5/v7 | 从正常运行文档移到归档/离线迁移区；删除前确认不再需要旧包迁移 |
| Proton旧shared路径别名 | proton_independent_v1/{charged,photon,schneider_joint.bin}为指向shared_independent_v1的符号链接 | 当前默认配置已使用shared路径；清理历史脚本引用后可去掉别名；这不会省出大块磁盘空间 |
| 老的proton专用批次包装 | scripts/run_ct_batch.py生成多份YAML再走plan-manifest | 默认文档只保留通用原生批次入口；显式逐批种子/审计元数据用途若需保留，迁移到通用工具后再删除包装脚本 |
| README历史叙述 | 仍写旧分支minibeam-minimal-20261001、7配置/初始打包结果及多轮旧吞吐；实际分支master | README只保留当前入口、支持范围和最新基准；历史结果集中docs/archive，保留failed.md及验证证据 |

这张表是删除候选和迁移前提，不是已完成的删除清单。其他用户YAML、benchmark生成器和回归脚本仍可能引用旧接口。

## 第二批：可以统一，但先要做迁移或验证

1. **出生包格式统一到MAIBIR02。** 实际proton_independent_v1/birth_bank.bin仍为MAIBIR01，carbon为MAIBIR02。后者增加显式Z/A；可对proton添加1/1标记，保持全部出生样本不变，更新SHA/清单并验证后，才删除MAIBIR01读取支持。不能现在只保留版本号较大的读取器。
2. **独立电子只保留precision=4运行入口。** precision=0参考运行仍存在；应从“独立charged路径”单独拆除，而不是全局禁用0，因为默认carbon compact及非独立路径也使用这个默认值。`generate_electron_fp32.py`仍从原始参考方程头文件生成FP32实现，磁盘读取器也使用原始结构；它们不等于可直接删除的旧运行后端。若要只留一份方程实现，需先调整生成架构。
3. **清理历史精度与性能分支。** 默认dose FP32 ON、FP64 OFF；FP64 dose scorer、电子FP64 geometry/loss-gate实验入口均可纳入退出清单。已接纳的参数复用、速率证书、voxel cache等ON/OFF对照可收敛为固定最新行为，但删分支会影响GPU编译布局，需复测。未启用的性能候选需逐项对照failed.md；性能剖析/计时开关和失败审计本身仍有用途，不能一并删除。
4. **主粒子调度统一。** combined/split/compact均存在；carbon water minibeam仍默认combined。先在该场景验证compact，再删除其历史入口。CT proton专用stage11只对严格匹配配置可用，carbon及其他场景仍需通用实现；关闭对照开关不等于可以删通用核。
5. **收敛重复环境开关。** PRIMARY_SPLIT、PRIMARY_SPECIALIZE可收敛到YAML；已接纳的CT material index、deferred birth、retained physics tables可保留自动准入并取消旧ON/OFF对照入口。但FE队列预算、设备/显存/粒子数等运行参数以及CARBON_RUNTIME_BREAKDOWN不是版本兼容，应保留合适公开入口。
6. **几何与输出键统一。** phantom_length_mm/depth_bin_width_mm、准直器旧尺寸名及旧dose/LET输出键仍有使用。可迁移为统一网格与输出接口；iec61217目前仍用于carbon水束源，TOPAS patient placement当前用于RT07575 CT。坐标变换的删除需要保证实际源位置、方向与isocenter完全一致，不能只改标签。

## 当前不能当作废弃内容删除

- water_response_v2与两个水电子包：两个water minibeam预设仍依赖。water_response_v1本轮未见预设使用，可以单独退出；v2不能随之删除。
- carbon compact v7及其材料index/源身份依赖：默认carbon CT仍用。替换成独立carbon路径属于物理实现迁移，需覆盖材料/密度/能量和局部剂量验证；既有头部材料覆盖缺口尚未补齐。
- carbon broadbeam joint response：与shared独立电子包不是同一用途，不能互换。
- FE、Urban、核反应、带电次级、光子、中子及质量/能量/队列审计：名称或引入时间不能决定是否保留。
- `legacy_highland`、`legacy_clipped_gaussian`等字符串：某些是被全局Urban覆盖的占位或仅针对另一粒子类型的选项，某些则可实际选择物理分支；需按有效路径收敛，不能批量字符串替换成urban_v2。
- 原始物理方程、制包来源、当前共享包和原生TOPAS对照数据。去掉兼容运行入口不等于去掉可追溯的验证基础。

建议执行顺序：先清理无效键/别名/README及历史入口，再统一数据格式和构建开关，最后分别迁移carbon CT、水minibeam及源几何。当前目标可设为“八个默认场景各保留一条明确运行路径，carbon独立试验单列；不再默默接受旧配置，不提供旧版自动回退”。不把静态删行数量推断为吞吐收益。

补充：逐场景物理包、电子包及旧数据退出清单见[八场景包映射](eight_scenarios_packages_20261004.md)。其中旧carbon迁移脚本参与material_prepared_sources构建指纹，迁移/删除时必须同步处理派生包身份；不能直接从构建目录移走。
