# 生产源码中的不可达诊断清理（2026-10-14）

本次删除已在生产构建中固定关闭、或已无配置入口的诊断实现，目标是减少维护成本，不涉及吞吐优化或物理模型近似。

删除范围：

- 固定关闭的粒子来源、同位素、微束组件与 ROI 计分，以及主粒子体素注量、出生谱缓冲区和结果字段。
- 铜逐 history 诊断、history mask、响应捕获文件，以及对应的读取、分配、记录和输出实现。生产铜反应子任务结构独立保留。
- 无法启用的微束相空间、带电碎片相空间与水中平面记录。
- 固定关闭的 EM 查表性能诊断、次级步进 profiling 和 FE 抽样计数；移除无调用的空 profiling 函数和局部审计 flush。
- 无实际条件编译引用的 `CARBON_HAS_SYCL`、`CARBON_DEFINE_SYCL_CONTEXT`，及相关过时注释。

`enable_charged_origin_voxel_scoring` 和 `minibeam_phase_space_at_copper_exit` 从配置接口及发布 YAML 中移除。以前的 `false` 不再作为兼容入口；使用这些键会在启动时被拒绝。固定 compact 调度的校验和强制步长覆盖保留。

保留生产能量账目、EM 有效性、核反应与弹性覆盖、带电物种能量及终止原因、完成计数和零溢出审计。电子/光子运输仍可显式启用；四条专用管线及 generic 管线保留。数据包格式、依赖和物理校准不变。

验证记录位于 `tmp/dead_code_cleanup_20261014/`：使用明确的 `sm_75` AOT 构建，并在 `CUDA_DISABLE_PTX_JIT=1` 下进行清理前后的匹配实现检查。这些检查不构成新的 TOPAS 验收或性能结论。

已完成的检查：

- 全部五条管线的 `sm_75` AOT 构建通过；所有 GPU 检查均禁用 PTX JIT。
- 八个发布预设分别进行清理前后各两批、每批 20,000 个主粒子的匹配回归。源分配与离散审计计数一致，最大剂量相对 L1 差异为 `3.25×10⁻⁷`；能量、完成和零溢出检查通过。
- 通用质子水宽束、通用碳离子水微束，以及显式启用电子/光子运输的质子 CT 微束和碳离子 CT 宽束匹配检查通过。
- 90 项配置/命令行接口检查、4 项核反应包契约检查、八预设数据清单及发布输入检查通过。
- 更新当前源码清单及文档中的示例配置，移除已删除文件和配置键的引用。

主要结果文件为 `eight_presets/summary.json`、`extra_regression.json`、`interfaces_check.json`、`package_contracts_check.json`、`data_manifest_check.json` 和 `release_inputs_check.json`。碳宽束电子补充检查使用显式电子包路径及 tracking 开关；最初缺少包路径的临时测试配置被启动校验拒绝，补齐后通过。

## 第二轮：编译器可消除的遗留源码

进一步删除：

- 始终以 `std::false_type` 调用的电子出生区域诊断，包括 lane 私有数组、核函数外层模板封装、空 GPU 指针、结果输出及释放代码。
- `secondary_tail_diag=false` 的日志分支；未使用的次级物种边界变量、切片参数和核函数物种模板参数；固定为一个元素的 active-count 容量条件表达式。
- 主粒子旧计分使用的能损分量副本、历史 stopping/density 副本、无消费者的查表索引与方向绝对值；次级未使用的查询能量和 target-A。
- continuous26 中恒真条件后的旧密度节点插值、非 continuous 的 range/rate/prepare 分支，以及无消费者的 section 目录 GPU 分配。保留当前节点与样条包，continuous26 仍使用这些数据进行能量插值。
- 能损函数未使用的 clock、proposal-rate 和 distance 参数，以及 continuous26 时钟更新未使用的第二个 factor 参数。实际步进时钟、consume/update 和断点续跑状态保留。

未使用的 `density_u` 改为 `static_cast<void>(uniform())`，保留抽样调用和随机序列。电子出生仍使用步开始时的 `energy_MeVu`；输运能量、物理计分和当前数值校准不变。共享铜碎片步进中的终止核反应账目变量仍用于其他实例，因此不按单个实例的 unused 提示删除。

第二轮验证文件位于 `tmp/dead_code_cleanup_round2_20261014/`。数值组件检查覆盖 145,600 组条件，能损字段与 RNG 精确一致，invalid 为零。

第二轮最终检查全部通过：全五管线 `sm_75` AOT 构建、八预设清理前后各两批 20,000 主粒子的匹配回归、两项 generic 检查、质子/碳离子电子开启检查，以及 90 项接口和 4 项包契约检查。八预设源与离散审计计数一致，最大剂量相对 L1 差异为 `3.02e-07`；运行能量、完成和零溢出审计通过。全部 GPU 检查禁用 JIT。

## 第三轮：单节点 continuous26 状态与单事件调度

- 删除重复的高/低密度节点状态、无用途的插值权重、恒等 `mix()` 和未赋值的 host-record 入口。当前状态只保留一个材料索引及密度缩放；`UnifiedEmState::point()` 表示当前 continuous26 材料视图。
- 步开始的 EM 参数直接使用一份 `UnifiedEmPointStep`，移除两份相同参数组成的 `UnifiedEmStep` 包装；主粒子、次级、电子出生及组件检查同步使用单份字段。
- 时钟只保留一个 rate。原来的 `(rate0+rate1)>0` 保留为 `(rate+rate)>0`，维持原浮点求值；实际光学深度时钟与续跑状态保留。
- 删除无调用的 native-mean/state-rate 辅助接口、只写不读的 device material-count，以及不再使用的搜索索引输出/复用参数。
- 删除先插值、随后被 continuous26 原地覆盖的旧 delta moments 中间结果及其 GPU 上传。原 delta moments 数据包、主机加载校验、依赖清单及校准保持不变。
- 次级每轮只提交一个 kernel，改为保存、等待和计时一个事件，移除三元素事件数组、提交计数及遍历。

第三轮检查记录位于 `tmp/dead_code_cleanup_round3_20261014/`。数值组件检查覆盖 145,600 组条件，prepared 与普通路径的能损字段和 RNG 精确一致，invalid 为零。

第三轮最终检查全部通过：全五管线 `sm_75` AOT 构建、八预设匹配回归、两项 generic 检查、质子/碳离子电子开启检查、90 项接口及 4 项包契约检查。八预设源与离散审计计数一致，最大剂量相对 L1 差异为 `3.03e-07`，能量、完成和零溢出审计通过。所有 GPU 检查禁用 PTX JIT。

## 第四轮：无消费者缓存及辅助接口

- 删除旧密度节点插值使用的共享网格标志、计算函数及启动日志。当前 continuous26 不读取该标志；包内网格及完整解析校验保留。
- 删除无人调用的 factor/range/mean/step 辅助链，以及没有调用者的第五组搜索索引。保留实际使用的节点、阻止本领、射程和反射程索引；包内四类样条仍完整读取和校验。
- delta moments 继续作为必须校验的 sidecar，校验全部头字段、长度及每个数值的有限性和非负性。改为固定大小分块读取，移除校验后无人读取的数值缓存、prefetch 结果包装及空计时阶段。
- 删除只写不读的 package node-count 字段，以及缺失数据报错中指向已不存在生成脚本的提示。缓存身份仍包含 EM 包和 delta sidecar 的路径、大小及修改时间。

第四轮检查记录位于 `tmp/dead_code_cleanup_round4_20261014/`。新增包契约检查覆盖 delta sidecar 的错误 magic、版本、节点数、截断、缺失，以及末尾的负值、NaN 和无穷大。

第四轮最终检查全部通过：全五管线 `sm_75` AOT 构建、八预设匹配回归、两项 generic 检查、质子/碳离子电子开启检查、90 项接口及 12 项包契约检查。八预设各版本均运行两批、每批 20,000 个主粒子；源与离散审计计数一致，最大剂量相对 L1 差异为 `3.00e-07`，能量、完成和零溢出审计通过。数值组件检查的 145,600 组能损字段与 RNG 精确一致，invalid 为零。全部 GPU 检查禁用 PTX JIT。

第三、四轮合计，相对提交 `cf1b3bc` 的核心 C++ 源码净减少 92 行，其中非空、非纯注释代码减少 81 行。当前 `include/` 与 `src/` 共 141 个 C++ 文件、31,660 总行、27,963 代码行；统计排除 scripts、extensions、构建及临时文件。

## 第五轮：固定 FP32 与未使用的管线常量

- 剂量原子操作固定为 FP32，删除不可达的 FP64 backend 标签分支及仅用于此条件的布尔常量。主机结果仍使用 double。
- 删除次级 FE 散射中声明后从未使用的 `two_pi`、`theta_scat` 和 `phi_scat`。
- 删除四个专用管线中无人读取的 `source_z/source_a` 静态常量。实际配置的源身份、管线路由及粒子校验保留。
- 修正剂量类型文件中已经不存在的 `CARBON_DOSE_FP32` 开关、旧模型和 FP64 诊断构建注释。

第五轮检查记录位于 `tmp/dead_code_cleanup_round5_20261014/`，使用第四轮程序作为匹配对照。

第五轮全五管线 `sm_75` AOT 构建、八预设匹配回归、两项 generic 及两项电子开启检查全部通过。每个版本、每项配置运行两批、每批 20,000 个主粒子；源分配及离散审计计数一致，八预设最大剂量相对 L1 差异为 `2.98e-07`，能量、完成和零溢出检查通过。全部 GPU 运行禁用 PTX JIT。本轮净减少 22 总行、16 代码行；当前核心 C++ 共 31,638 总行、27,947 代码行。

## 第六轮：全面静态审计后的无调用实现

- 删除 33 个无调用函数，包括旧 copper Urban 辅助接口、未使用的 delta sampler、旧计分分类、矩形准直器辅助链、未使用的整条 polyline bounds/包装函数和 companion manifest key helper。当前圆柱准直器、charged delta sampler、Urban limiter、indexed polyline replay 与实际输入缓存保留。
- 删除闲置的 `Particle1D`、`SchneiderTargetDiagnostics`、`NoExpectedScore`，以及仅服务已删除函数的 bounds/true-path 类型。
- 删除旧 `TransportProfile` 模块、结果字段和编译文件项；CT 面距离函数移除诊断输出参数及分支，调用点与组件检查同步更新。CT 面距离计算本身保持原样。
- 删除六个无人读取的旧输出字段及 clear 语句、只写不读的三个 CT 原点副本、闲置自动调优标志和缓存累计计数。保留 MHD 输出、CCTG 原点及实际缓存命中判断。
- 删除电子 voxel cache 中无消费者的诊断宏和计数；保留显式电子运输使用的缓存。
- 更新源码清单，移除被删除文件的引用。保留 live ion-species 审计枚举及其数值、当前数据依赖清单与辐射长度参考资产。

第六轮记录位于 `tmp/dead_code_cleanup_round6_20261014/`，使用第五轮程序作为匹配对照。本轮核心 C++ 净减少 880 总行。

第六轮最终检查通过：全五管线 `sm_75` AOT 构建、八预设各两批 20,000 主粒子的匹配回归、两项 generic 及两项电子开启检查、90 项接口检查和 15 项发布输入/质量约定/TPS/CT 几何检查。八预设源分配与离散审计计数一致，最大剂量相对 L1 差异为 `3.07e-07`，能量、完成和零溢出审计通过。所有 GPU 运行禁用 JIT。

当前核心 C++ 为 138 个文件、30,758 总行、27,170 代码行；本轮减少 880 总行、777 代码行。首次编译暴露一处仍传入已删除诊断参数的调用点，修正后最终构建及全部检查通过。

## 第七轮：无入口主机接口与孤立辅助链

- 删除无调用的 C++ 核反应包 writer/builder、主机 duplicate lookup/fixed replay、旧 getter 和随之孤立的 builder helper。保留 CINPKG04 reader、CRC/energy-index 验证、channel domain、device table 与设备查找；extensions 中实际用于提取的 Python writer 保留。
- 删除 CT 二进制写出、主机 target sampling/总率插值、主机 CSDA、IDD/Bragg 分析等无人调用的接口和声明。源/核反应参考 stopping 表、输入校验及实际 package 加载保持不变。
- 删除已无外部调用者的 `longitudinal_ray.hpp` 整条辅助链，更新 include 与源码清单。
- 删除闲置的结果/缓存类型、旧结果字段及只服务它们的常量和注释。`domain_retained_MeV` 虽无读取者，仍由 aggregate assignment 写入，移除时同步去掉对应主机赋值项；有效 electron diagnostics 与设备计数布局保留。
- 删除无消费者的 `charge_power` 计算、`range_branch` 输出字段和 host cumulative-range 缓存。实际能损 range 分支、能量计算、stopped 判断及 RNG 不变。
- 保留二进制布局字段、live ion-species/EM/coverage/completion/overflow 审计、源归一化及当前依赖清单。

第七轮记录位于 `tmp/dead_code_cleanup_round7_20261014/`，使用第六轮程序作为匹配对照。数值组件检查覆盖 145,600 组条件，能损字段与 RNG 精确一致，invalid 为零。本轮核心 C++ 净减少 1,204 总行、1,034 代码行。

第七轮最终检查全部通过：全五管线 `sm_75` AOT 构建、八预设各两批 20,000 主粒子的匹配回归、两项 generic 和两项电子开启检查、90 项接口、12 项包契约和 15 项发布输入检查。八预设源分配与离散审计计数一致，最大剂量相对 L1 差异为 `3.02e-07`，能量、完成和零溢出检查通过。所有 GPU 运行禁用 JIT。

当前核心 C++ 共 137 个文件、29,554 总行、26,136 代码行。相对提交 `cf1b3bc`，尚未提交的第三至七轮合计净减少 2,198 总行、1,908 代码行。

## 第八轮：残留无调用函数及无消费者状态

- 删除残留的 C++ 写包辅助函数、构建索引辅助函数和整条无调用的主机 partial-rate 插值链；运行时 GPU 率插值、节点读取与域验证不变。
- 删除核反应表中三个只写不读的头信息副本；CINPKG04 文件头和序列化记录布局不变。
- 删除内部 GPU 紧凑交互记录的三个无人读取的 parent-local 方向字段及其转换计算。读取阶段原有 collision/parent 方向有限性和单位长度检查保留；它们已保证所删转换的归一化前提。
- 删除没有读取者的 `ct_schneider_radiation_length_file` 配置状态与路径解析，依赖清单不再将参考 JSON 列为八预设的运行必需文件。实际 Schneider 辐射长度常量和安装的参考资产保留；更新 data manifest 的预设列表。
- 现有接口契约检查加入已删除配置键，确保运行时拒绝该键。

本轮检查记录位于 `tmp/dead_code_cleanup_round8_20261015/`；核心 C++ 净减少 114 总行、105 代码行。当前共 137 个文件、29,440 总行、26,031 代码行。全五管线 `sm_75` AOT 构建、八预设各两批 20,000 主粒子的匹配回归、两项 generic 和两项电子开启检查全部通过；八预设最大剂量相对 L1 差异为 `3.15e-07`，源分配与离散审计计数一致，能量、完成和零溢出检查通过。91 项接口、12 项包契约、15 项发布输入检查及 173 项安装资产/八预设数据清单校验通过。全部 GPU 运行禁用 PTX JIT。
