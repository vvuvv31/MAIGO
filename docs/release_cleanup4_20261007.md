# 发布兼容代码清理：第四轮（2026-10-07）

本轮清理上一轮审计确认的旧版本分支、无调用的设备助手、废弃记账字段，以及水模体 Z 几何别名。未更改输入数据包或当前物理模型。

## 代码变化

- `run_quality.cpp` 删除 v1 默认文件回退、版本探测及非 v3 判定分支，直接从必需的当前 physics bundle 读取项目粒子注册表。保留 bundle 范围内的不支持粒子检查、日志截断拒绝、Be6 既有策略、目标覆盖、post-EM null collision 和守恒检查。缺失 bundle 记录失败，无法读取 bundle 拒绝审计。
- `io.cpp` 删除 `v1-13` provenance 输出分支。
- `secondary_rate_table.hpp` 删除 v1 网格和旧 13-projectile 常量，空表不再带旧网格默认值；当前 v3 对象使用现有 921 节点网格。
- 删除整个无调用方的 `src/detail/sycl_score_device.inc` 及 include：指数深度沉积、LET 限制记分、旧插值/中点阻止本领、出生记分助手均无调用。不修改有效的 GPU 记分路径。
- 删除旧八项账本中的七个无读取字段、无调用的 `total_accounted_MeV()`，以及始终为零的中性/不支持带电粒子摘要。保留实际核产物能量 sink、核反应分项账本和全局物理闭合。
- 源码净减少 433 行；更新源码清单、预设依赖清单、README 和 AGENTS。

## 几何配置迁移

仅接受 voxel 网格作为几何来源，`phantom_length_mm` 和 `depth_bin_width_mm` YAML 键退休，任何传入均拒绝。水模体要求明确的 `voxel_bins_z` 和 `voxel_size_z_mm`，不再通过旧长度字段计算层数。CT 从 CCTG 头读取几何，显式 voxel 参数作为一致性断言；原 CT 浮点 extent 的计算保持一致。输运内部长度和层厚从规范几何派生，保留必要的内部缓存。

| 水预设 | Z 层数 | 层厚 mm | 长度 mm |
|---|---:|---:|---:|
| proton_water_broadbeam | 460 | 0.5 | 230 |
| proton_water_minibeam | 460 | 0.5 | 230 |
| carbon_water_broadbeam | 400 | 0.5 | 200 |
| carbon_water_minibeam | 1000 | 0.25 | 250 |

四个水模体的层厚、长度与冻结配置完全一致。碳水 minibeam 由旧长度/层厚配置改成明确的 1000×0.25 mm 网格。

## 验证

- 最终 `BUILD_JOBS=6 bash scripts/build.sh` 成功（`release4_final_build_20261007.log`），现有编译警告仍在。
- 60 项接口拒绝/必需字段检查通过（`release4_final_interfaces_20261007.json`）；包括退休的两个几何别名、缺失及零值 Z 网格。
- 12 项严格输入/质量约定检查通过（`release4_inputs_20261007.json`）。
- 4 项 primary 包契约检查通过（`release4_final_contracts_20261007.json`）。
- 最终八预设 GPU 对照回归均通过（`validation_release4_final_20261007/summary.json`）；每个预设每个臂连续 2 批，每批 10,000 histories。源 CSV/种子和离散 EM/电子/光子/Urban/核反应计数一致，覆盖/完成检查通过，所有队列零溢出。
- 最大剂量相对 L1 差异 `2.90944516947e-07`（阈值 `1e-5`）；16 个候选批次最大物理相对能量残差 `7.26584448022e-06`（阈值 `1e-4`）。
- 源码清单 126 个文件、输入清单及八预设依赖清单核对通过；已删除的符号/旧路径在运行源码中无残留。

冻结基线位于 `/tmp/maigo_cleanup4_before_20261007/`，包含未修改的八个原始配置、源码和二进制；此基线已经移除了中性粒子追踪。本轮对照无需调整配置。对照/候选 SHA256 见 `release4_control_20261007.json`。

第一轮回归 `validation_release4_20261007/` 是删除整个闲置助手文件前的检查；以最终回归记录为准。已有历史报告未改写。

这是匹配 GPU 实现回归，不是新 TOPAS 剂量验收或性能测量；碳 minibeam 剂量精度验收仍待完成。

当前复现构建请显式使用 `BUILD_JOBS=6 bash scripts/build.sh -DCARBON_CUDA_ARCH=sm_75`
（根据自己显卡设置：2080 Ti=`sm_75`，A6000=`sm_86`，L4=`sm_89`），
运行前设置 `export CUDA_DISABLE_PTX_JIT=1`。上述历史构建记录保留原命令。
