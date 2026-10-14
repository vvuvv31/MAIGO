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
