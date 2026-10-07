# 中性核产物追踪清理（2026-10-07）

用户要求删除中性粒子追踪。本次范围为中子及中性核产物追踪；电子电磁级联中的光子输运保留。

## 删除内容

- 删除 `src/neutron_package.cpp`、`src/neutron_transport_sycl.cpp` 及对应两个头文件，移除 CMake 源文件入口。
- 删除 `data/neutron/` 的 7 个文件，共 120,072,470 字节（约 114.51 MiB），包含 native collision/macroscopic 和 recoil stopping 数据包。
- 删除配置字段、白名单入口、加载/校验/预取、中子出生缓冲及捕获、GPU 追踪调用、随机流分支标签。
- 删除中性粒子队列及废弃摘要结构、追踪计数/耗时/剂量图、实验诊断写出和旧零值输出字段。
- 删除未使用的 neutral kerma 比例助手；不引入局部沉积替代。
- 更新 README、AGENTS、源码/输入清单及八预设依赖清单。旧验证报告和输出是历史证据，未改写。

## 保留的物理行为

核反应包仍识别 neutron/gamma 产物，用于包契约和显式未追踪能量记账。这些标识及能量 sink 不代表追踪功能。中性能量不会增加剂量。带电离子、碎片、电子及电磁光子输运、FP32 记分、涨落、步长、compact 调度和批次/种子行为保留。

质子水 minibeam 原默认开启实验中子追踪，现在不再包含该剂量贡献。这是用户要求的物理范围变化；不能把本次结果宣称为与中子开启基线等价。

## 验证

- `BUILD_JOBS=6 bash scripts/build.sh` 成功，日志 `no_neutral_build_20261007.log`；现有编译警告仍在。
- 严格输入检查 12 项通过（`no_neutral_inputs_20261007.json`）。
- 旧接口拒绝检查 54 项通过，包含 13 个退休的中子/中性输出配置键（`no_neutral_interfaces_20261007.json`）。
- primary `has_support` 包契约 4 项通过（`no_neutral_contracts_20261007.json`）。已删除中子包检查。
- 八预设匹配 GPU 回归均通过；每个预设每个对照臂连续两批，每批 10,000 histories。源 CSV/种子、离散 EM/电子/光子/Urban/核反应审计一致，能量/覆盖/完成审计通过，所有队列零溢出；未生成中子追踪文件或退休字段。
- 最大剂量相对 L1 差异 `2.76020893905e-07`，阈值 `1e-5`；候选 16 批最大物理相对能量残差 `7.26648467937e-06`，阈值 `1e-4`。
- 源码/输入/依赖清单已核对；新静态库无中子追踪符号；旧构建目录中子对象已删除。

冻结对照 `/tmp/maigo_no_neutral_before_20261007/carbon_mc`；仅将其质子水 minibeam 的 `enable_neutron_transport` 改为 `false`，其余对照预设不变。原始源码、二进制、开启中子的配置和中子包备份位于同一外部临时目录。对照/候选 SHA256 见 `no_neutral_control_20261007.json`，回归见 `validation_no_neutral_20261007/summary.json`。

本次是关闭中子追踪基线下的实现回归，不是新 TOPAS 剂量验收或性能测量；碳离子 minibeam 剂量精度验收仍待完成。
