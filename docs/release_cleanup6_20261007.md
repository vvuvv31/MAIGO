# 发布兼容代码清理：第六轮（2026-10-07）

运行和构建入口收敛为八个 CUDA 预设，删除旧后端适配及无调用的类型/API。配置、数据包和物理模型保持一致。

## 改动

- `device.cpp` 只接受 `cuda`，删除通用 GPU/CPU/default 入口、OpenCL/Level Zero 路由、其他 backend 描述及 vendor 别名，删除 CPU 不可用时转 GPU 的回退。队列仍启用 profiling 和 in_order，CUDA 设备排序分数保持原值。
- 删除无调用的 `probe_sycl_device()`、`DeviceCapabilities` 及通用 backend 命名/匹配 helpers；有效设备描述保留。
- `TransportConfig::validate()`、队列创建及输运入口明确拒绝非 `cuda` 值。删除空设备名转通用 GPU 的路径；输运对实际队列后端和 GPU 类型进行检查。
- 删除 `CARBON_PRIMARY_CONTEXT_POINTER` / `CARBON_PRIMARY_CONTEXT_MOVE_UNIFIED`、设备 context 指针的分配/上传/捕获分支以及 CMake 定义。保留原 CUDA 按值捕获。
- CUDA local size 保持 128；默认 histories chunk 保持 >1M 时 34816，否则 broadbeam 17408、minibeam 4096。显式 `history_chunk_size` 仍有效。删除 Intel/其他后端的分支。
- 删除无调用的 `MscModel`、`MscStepResult`。Highland、Fermi-Eyges、Urban 的有效物理实现保留。
- 删除入口白名单之后不可达的旧键单独检查，统一由白名单拒绝；增加这些键的接口回归。
- CMake 只接受 `nvptx64-nvidia-cuda` target，CUDA architecture 配置保持支持。更新 README、AGENTS。

## 验证

- 构建成功（`release6_build_20261007.log`）；现有编译警告仍在。
- 82 项接口检查通过（`release6_interfaces_20261007.json`），包括 11 个退休设备值/空值及 11 个退休配置键。
- 15 项输入契约检查通过（`release6_inputs_20261007.json`）。
- 4 项核反应包契约检查通过（`release6_contracts_20261007.json`）。
- 非 CUDA 构建目标 `spir64` 被明确拒绝（`release6_build_contract_20261007.json`）。
- 八预设匹配 GPU 回归通过（`validation_release6_20261007/summary.json`）：每个预设每个臂连续 2 批，每批 10,000 histories。源 CSV/种子与离散 EM/电子/光子/Urban/核反应计数一致，能量、覆盖和完成审计通过，所有队列零溢出。
- 最大剂量相对 L1 差异 `2.77674195144e-07`（阈值 `1e-5`）；16 个候选批次最大物理相对能量残差 `7.26611022326e-06`（阈值 `1e-4`）。
- 源码清单 126 个文件、输入及八预设依赖清单核对通过；已删除的适配宏、设备探测 API、旧类型和后端分支无运行源码残留。

冻结对照 `/tmp/maigo_cleanup6_before_20261007/` 包含未修改的八个 CUDA 配置、源码和二进制；对照/候选 SHA256 见 `release6_control_20261007.json`。旧报告保持原样。

本次为匹配 GPU 实现回归，不是性能测量或新的 TOPAS 剂量验收；碳 minibeam 剂量精度验收仍待完成。
