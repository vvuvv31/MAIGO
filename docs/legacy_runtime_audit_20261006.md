# 当前默认路径的旧兼容代码审查

日期：2026-10-06。范围：八个当前YAML、配置展开/校验、SYCL输运、电子包加载和相关主机诊断。静态阅读并核对上一轮八预设smoke文件访问记录；本轮未做新的蒙卡、性能实验或代码删除。RT07575为头部病例，默认CT minibeam肺CT另行区分。

## 结论

默认包已切换为26组成，但生产电子加载器仍保留可显式选择的旧出生、83节点charged/photon以及混合26/83实验接口。因此“默认不读取旧包”已经成立，“运行程序只支持新包”还没有成立。这不是当前默认路径偷偷回退，而是仍能通过旧文件/环境变量选择旧实现。

离子175节点的情况不同：SYCL设备视图强制continuous26；旧节点插值主要是主机诊断/归档对照接口，不应误称当前GPU仍做175节点查询。

## 优先清理项

| 优先级 | 路径/接口 | 证据与当前用途 | 建议 |
|---|---|---|---|
| 1 | electron_fp32_device.hpp 的 complete26/else；input_dependencies.hpp 双索引接受 | 根据charged magic选择MAIEC026或MAIECB02；photon接受MAIPH026或MAIPHB02。默认用26，旧索引仍可显式进入同一加载器 | 生产构造器严格限定MAIEC026/MAIPH026各26组成；旧读取器移至显式offline/reference工具 |
| 1 | 旧出生bank及出生准备/抽样分支 | electron_fp32_device.hpp读取MAIBCA26，否则ElectronBirthBankDevice；transport_sycl.cpp中cut_birth26真假决定新旧出生准备和worker抽样 | 默认生产必需MAIBCA26，删除生产侧旧出生转换、GPU视图和运行分支；保留离线出生参考 |
| 1 | 混合26硬率/Moller＋83终态路径 | CARBON_ELECTRON_RATE26_FILE、CARBON_ELECTRON_MOLLER26；electron_fp32.hpp允许无composition26但有rate26/moller26，低能继续旧终态 | 归档hybrid实验，生产只保留完整26碰撞；不能删除完整26仍在用的NativeGridRateView/rates26.bin |
| 1 | charged密度双节点查找/混合 | electron_fp32.hpp的charged_density_model、charged_parameters、charged_inverse_range、sample_charged_collision和density_bracket都有新旧分支；旧分支持lower/upper/fraction和empirical终态 | 将生产类型改为单composition＋实际density；去掉旧两节点参数/终态混合和不再需要的字段，离线对照另行编译 |
| 2 | photon密度节点混合及旧native终态格式 | 完整26按composition_only使用一组标定条件终态；通用FP32代码和PhotonResponseData仍接受旧密度索引/MAIPHN01 native记录 | 生产限定当前MAIPHF01组成记录，移出旧密度混合；保留现有标定密度/实际cut近似，不能把存储清理当物理修正 |
| 2 | 已退休核模型配置字段/重复校验 | config.cpp前段拒绝nuclear_model=cinel02，后段还列为合法枚举并校验旧primary/water/CINEL02文件；TransportConfig仍有这些字段及解析 | 去掉这些旧模型字段、解析和不可达校验，留下明确的旧键拒绝错误。现役cinel02_max_secondary_inelastic_generations需统一改名或暂留，不可误删功能 |
| 2 | CCTG v1四类材料、旧CT CSV核截面分支 | ct_grid.cpp接受v1/v2/v3；config.cpp仍有legacy-four-class及Schneider CSV预加载。四个当前CT YAML实际输入头均为CCTG v3，核反应走v3 bundle | 生产限定当前25-Schneider CT契约；v1/四类转换与旧CSV读取移至离线转换。是否保留v2需检查外部DICOM打包/转换用途 |
| 2 | 旧YAML别名 | tps_source/tpsSource、tps_gantry_angle_deg/tps_beam_angle_deg、ct_secondary_exact_faces_diagnostic覆盖正式字段、topas_patient_rot_z | 固定公开键并移除别名；先迁移运行脚本和full-plan生成器。tpsSource本身是当前preset使用键，应保留或同步迁移 |
| 3 | 旧电子double参考transport/upload工具 | electron_charged_bank_device.hpp、photon_response_bank_device.hpp、electron_birth_bank_device.hpp及其host参考算子主要承载旧格式/独立GPU诊断；完整26加载器仍include这些头 | 在去掉生产旧分支后解除include依赖，移到reference目录。保留仍被当前joint/photon加载共用的结构和辅助函数，不能整头盲删 |
| 3 | 仅脚本调用的电子原型 | electron_analytic_birth.hpp、electron_response26.hpp等被独立组件检查脚本引用，不是完整26主输运接口；electron_elastic_batch.hpp还包含electron_batch_original.hpp | 合并到离线实验/参考区域，并核对间接include。文件名含26不等于当前默认26模型 |
| 3 | 调度/probe及精度实验开关 | CMake和transport_sycl.cpp保留species probes、context候选、primary split/combined/compact及FP64 geometry/loss选项 | 与旧物理兼容分开处理；归档已失败候选，保留剖析/诊断开关。当前Carbon water minibeam没有显式compact，走默认combined，不能直接删combined |
| 3 | 旧审计说明 | run_quality.cpp仍说density-node mixing、C12 birth-only extraction及CT dose pending；当前完整26已更换出生与碰撞 | 更新说明以准确记录26近似和分场景验收；保留coverage、能量、completion、queue等审计 |

代码定位：

- [生产电子加载](../include/carbon/electron_fp32_device.hpp)、[输入身份缓存](../include/carbon/input_dependencies.hpp)。
- [FP32电子算法与旧节点分支](../include/carbon/electron_fp32.hpp)、[主输运出生分支](../src/transport_sycl.cpp)。
- [配置解析/校验](../src/config.cpp)、[配置字段](../include/carbon/transport_config.hpp)、[CT加载](../src/ct_grid.cpp)。
- [旧charged载荷读取](../include/carbon/electron_charged_data.hpp)、[photon载荷读取](../include/carbon/photon_response_data.hpp)。

## 不应按旧名称直接删除的内容

1. **inelastic_package_v2.cpp/hpp和sycl_cinel02_device.inc**：名称老，但文件实际解析CINPKG03、提供现役反应事件结构、查找和能量台账。src/inelastic_package_v2.cpp要求包version3；不能当成仍在使用CINEL02物理。旧无全局索引头的兼容分支是否能删，需要逐一检查当前铜包header flags。
2. **Carbon铜legacy_clipped_gaussian**：carbon_ct_minibeam显式选中，另一carbon minibeam依赖默认。它是当前物理配置的一部分，替换属于改变铜能损模型，必须单独验证，不能当死代码清掉。
3. **iec61217**：carbon_water_minibeam.yaml正在使用。dicom_lps是CT计划首选，但不能直接删水源所需坐标模式。
4. **TOPAS spot解析、spot_weights_file及minibeam_pbs_y**：八preset之外，最新RT07575头部full-plan正式配置实际使用这些接口；删除会破坏full-plan重跑。可移除的是已无用户的pose适配分支，先检查生成器和历史/当前源场景。
5. **joint.bin及其回退到独立电子步进**：这是现役空间扩散加速和异质界面处理，不是回退旧82节点。25组成joint保留，Water_75eV没有joint，必须保留独立步进。
6. **离子EM主机旧读取接口**：unified_em_view.hpp的__SYCL_DEVICE_ONLY__强制continuous26，设备不进入旧密度目录。归档验证需要旧主机读取，可隔离到reference构建；continuous标量内部的小密度/cut网格不是175个完整响应状态。
7. **transport_cpu.cpp**：serial输运只剩拒绝stub，但同文件还含能量平衡、IDD和Bragg指标，被当前结果处理共用。最多移走公共统计后清掉stub，不能删除整个文件。
8. **FP64参考/开关及诊断**：默认全部FP32，关闭的编译分支不代表运行开销，也不能把普通host double统计误称GPU FP64输运。若只分发FP32生产，可将参考选项拆分；本次未配置或运行FP64。

## 推荐执行顺序与验收

先收紧生产包schema和去除旧出生/hybrid入口，再简化电子GPU视图及节点混合，之后清理退休配置/CT格式；离线参考最后独立组织。同步修正日志和Materials and methods。

每阶段必须保留新包覆盖、能量、完成、零溢出审计，重跑八默认preset的FP32 smoke和RT07575头部小计划。改变抽样顺序/状态布局或铜物理时，要按现行CT剂量标准独立对照：broadbeam local1%/1mm>99%，search0.1mm；proton minibeam替包gamma下降≤1个百分点，search0.02mm，绝对结果单独报告；carbon minibeam准确性暂缓但运行审计必须过。

以上主要降低维护、编译和错误选择旧路径的风险；仅删除默认未执行/编译已消除的分支，不保证稳态吞吐提升。应另测编译时间、二进制/内核资源和同条件resident吞吐，不从代码行数推断加速。
