# 水／Schneider CT 的 Unified EM 清理

## 范围

用户确认清理水／CT 的 legacy 能损路径，保留独立模块。改动以本轮开始时的工作树为基准；没有用 Git HEAD 覆盖之前尚未提交的优化。

- 水／CT 主粒子和包覆盖的 18 种带电次级只使用 `g4_material_joint_v1`。
- 保留 FE、Highland、Urban 的 MCS 选择，以及核反应、带电次级和现有电子响应。
- 铜／空气束线、中子及未覆盖物种的独立弹性反冲模块保留。
- 水参考仍为 `Water_75eV`。没有调整 stopping scale 或重新拟合物理参数。

## 已删除

1. `EmMode` 模板参数及 `CARBON_EM_SPECIALIZE` 的 legacy 分派。
2. 水／CT 的旧主粒子 stopping、midpoint／CSDA 能损和旧涨落采样分支。
3. 已覆盖带电次级的旧 species／Schneider stopping 查询、表覆盖检查及 CSDA fallback。
4. 多余 stopping／CSDA 设备上传、host cache、启动 SHA 检查及 bundle stopping 依赖。
5. 独立旧涨落包、Schneider primary stopping、section-ion stopping 的头文件和源文件；旧 CT mass-SPR 和设备 CSDA 辅助函数。
6. 仅实现 legacy EM 的 serial 求解器；`transport_serial` 接口保留并明确报错。CPU 类型的 SYCL 设备并未因此禁止。
7. 维护中的 Unified YAML 中241 行旧选择／表路径／pin／scale 字段。

净删行数和变更文件清单以 `changed_files.json` 为准；当前净减少 **4304 行**。本轮配置清理仅修改含 `em_model: g4_material_joint_v1` 的配置，历史 evidence 输入不迁移。

## 配置行为

默认模型为 `g4_material_joint_v1`，必须提供 Unified EM 包和 SHA pin；delta moments 与核反应 rate/event 包的校验继续保留。

- `em_model: legacy`、`device: serial`、`enable_secondary_unified_em: false` 明确报错。
- 启用旧 CSDA／固定块涨落／Schneider stopping 诊断、选择独立旧涨落包、非单位旧 stopping／straggling scale 或非空旧 scale 表均报错。
- 旧 Unified 配置中的 `primary_em_model: legacy`、`enable_secondary_unified_em: true`、`ct_primary_midpoint_stopping`、C12 专用 Unified 开关及旧 stopping 路径／pin 作为无效兼容字段消除，不再出现在 canonical effective config 中。
- 旧核包 JSON 的 `stopping_table` 段可以保留归档信息；加载和后台 SHA prefetch 不访问它的数据文件。
- `primary_stopping_power_file` 接口仍提供独立核截面／反冲插值所用的参考能量网格；它的 stopping 值不再用于水／CT 主粒子能损。未覆盖弹性反冲仍使用独立参考表及此前的插值逻辑。
- CT 使用 Schneider section ID 网格，材料编号必须在 `[0,24]`。包内材料／能量覆盖与采样有效性由 Unified EM 检查。

## 已知行为变化

CT 主粒子的终止能量现在直接取 YAML `energy_cutoff_MeV`。旧路径还会把它提高到旧 Schneider 表最小 MeV/u × A，例如 C12 从 0.1 提到 0.12 MeV。删除旧表依赖后不再隐式提高这个值。新旧剂量严格一致不能只凭源码清理推断。

## 检查范围

- 本地 Slurm 编译，CUDA SYCL Release O3，4 CPU、24 GiB，`CARBON_DOSE_FP32=ON`、`CARBON_DOSE_FP64=OFF`、`BUILD_TESTING=OFF`，保留 FE joint queue 支持。
- 最终编译作业：**21153**；最终状态与二进制 SHA 见 `build_result.json`。
- 静态检查确认 secondary device context 的 131 个成员与位置初始化逐项对应。
- 核反应、Unified EM 采样器、Urban、电子队列及中子／反冲独立实现的 11 个关键文件与清理前逐字节一致，见 `source_review.json`。
- `git diff --check` 无空白错误。
- 没有运行新的输运测试、energy/EM 审计、质量检查、剂量／gamma 或吞吐 benchmark，也没有宣称性能提升。先前 167–170k/s 和 gamma 数据属于清理前二进制。
- 仍存在既有编译 warning；成功编译不代表所有运行配置通过物理验收。

## 文件和回退依据

- 清理前源码／配置：`/mnt/sda/wuwei/unified_em_cleanup_20260930/before/`
- 清理前二进制：`/mnt/sda/wuwei/unified_em_cleanup_20260930/baseline_carbon_mc`，SHA256 `109768dd6a61854a4cfe92257c9eba6d38f1d8b31f1315b2bc1431f4c52fcc98`
- 最终冻结源码：`/mnt/sda/wuwei/unified_em_cleanup_20260930/source/`
- 新构建：`/mnt/sda/wuwei/unified_em_cleanup_20260930/build/`；工作树入口 `build/unified-em-cleanup/carbon_mc`
- 编译参数和提交脚本：`build_args.json`、`build.sbatch`
- 本轮独立 patch：`incremental.patch`，基于清理前工作树快照，不混入已有改动。
- 原有已验证的 `build/fe-joint/carbon_mc` 保留。本轮未 commit／push。
