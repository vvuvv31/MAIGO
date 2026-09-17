# MAIGO 当前源码结构

更新日期：2026-09-05。本文件描述当前构建图，不复述历史设计。
构建权威是 [CMakeLists.txt](../CMakeLists.txt) 和 [presets](../CMakePresets.json)。

## 运行数据流

```text
config / spots / CT / verified physics data
                  ↓
main → config + plan_run + source transforms
                  ↓
carbon_core → transport_sycl.cpp（本地 SYCL GPU）
                  ↓
TransportResult → run_quality + io
                  ↓
3D DoseToMedium / ledger / quality → shard merge → Gamma
```

## 模块职责

| 文件或目录 | 当前职责 |
|---|---|
| [main.cpp](../src/main.cpp)、[cli.cpp](../src/cli.cpp)、[config.cpp](../src/config.cpp) | 命令行、轻量配置、数据加载与路由校验 |
| [plan_run.cpp](../src/plan_run.cpp) | spot 计划运行与聚合入口 |
| [topas_spots.cpp](../src/topas_spots.cpp)、[tps_source.cpp](../src/tps_source.cpp) | spots 输入、history 分配、源几何与患者变换 |
| [ct_grid.cpp](../src/ct_grid.cpp)、[ct_dicom.cpp](../src/ct_dicom.cpp) | Schneider HU 解析、连续密度、section、CCTG/DICOM |
| [transport_sycl.cpp](../src/transport_sycl.cpp) | 主/次级步进、队列、材料选择、核反应、剂量与诊断 |
| [transport_cpu.cpp](../src/transport_cpu.cpp) | 串行功能子集；不等于患者 GPU 路径 |
| [schneider_rate_table.cpp](../src/schneider_rate_table.cpp)、[secondary_rate_table.cpp](../src/secondary_rate_table.cpp)、[schneider_target_sampler.cpp](../src/schneider_target_sampler.cpp) | 材料/抛射体核反应率、域约束和靶选择 |
| [inelastic_package_v3.hpp](../include/carbon/inelastic_package_v3.hpp)、[inelastic_package_v3.cpp](../src/inelastic_package_v3.cpp) | CINEL03 主机/设备查询；文件实际为 CINPKG04 v4 |
| [inelastic_package_v2.cpp](../src/inelastic_package_v2.cpp) | 保留的 CINEL02 water 支持；不是 CT fallback |
| [schneider_stopping_table.cpp](../src/schneider_stopping_table.cpp)、[stopping_power.cpp](../src/stopping_power.cpp) | Schneider C12 与 water/ion stopping 数据 |
| [multiple_scattering.hpp](../include/carbon/multiple_scattering.hpp)、[straggling.hpp](../include/carbon/straggling.hpp)、[energy_loss_fluctuation.cpp](../src/energy_loss_fluctuation.cpp) | MCS、凝聚能损涨落及 water 数据路径 |
| [schneider_delta_tail.cpp](../src/schneider_delta_tail.cpp) | section-0 C12 电子剂量响应表；候选与已验证范围需分开 |
| [transport.hpp](../include/carbon/transport.hpp)、[transport_config.hpp](../include/carbon/transport_config.hpp) | 配置/结果、ledger、diagnostic 字段与枚举 |
| [run_quality.cpp](../src/run_quality.cpp)、[io.cpp](../src/io.cpp) | 质量门禁、MHD/RAW/JSON 及可选诊断输出 |
| [startup](../startup/README.md) / [extensions](../startup/extensions/README.md) | TOPAS 提取与扩展源码；本机实际 TOPAS 源码/构建在 /home/wuwei/topas |
| [tools](../tools)、[tests](../tests) | 数据编译/审计、运行/比较脚本与测试 |
| [data](../data/ACTIVE_DATA.md)、[evidence](../evidence)、[benchmark](../benchmark) | 运行数据、冻结证据、病例及结果入口 |

当前 CMake 在 SYCL 开启时编译 `src/transport_sycl.cpp`、`device.cpp`、
`transport_profile.cpp`。旧文档的独立 legacy/dispatch/minibeam kernel 图不适用。
`CARBON_ENABLE_MINIBEAM` 选项仍存在，但不能据此宣称旧文档中的专用双 kernel
结构存在或 minibeam 已验证。历史多离子/电子实验亦不属于本次 CT 验证承诺。

## 构建与测试

`carbon_core` 是核心库，`carbon_mc` 是主程序。默认 SYCL 关闭；
GPU 配置需 `CARBON_ENABLE_SYCL=ON`，NVIDIA target 为
`nvptx64-nvidia-cuda`，本机 arch 显式设为 `sm_75`。
Dose atomics 默认 FP32；本工作不通过精度类型调参。
`CARBON_VALIDATION_SCORERS` 是编译期开关，运行期输出配置是另一层；
不是开此选项就自动输出全部诊断，也不是关闭就没有 run-quality 检查。

`BUILD_TESTING=ON` 时当前注册 4 个 CTest：

1. `carbon_tests`
2. `inelastic_package_v2_tests`
3. `test_schneider_dicom_reference`
4. `test_secondary_rate_table_hardening`

另有 `run_step13_gpu`、`run_step19_gpu`、
`run_step20_gpu`、`run_step21_level3_gpu` 和 `dump_tps_phase_space`
等辅助 executable；它们不是每个都注册为 CTest。

```sh
ctest --test-dir build/oneapi-nvidia-release --output-on-failure
python3 tools/verify_schneider_v2_1_data.py
python3 -m unittest discover -s tests -p 'test_topas10x*.py'
```

注意：数据清理把部分旧测试 fixtures 移入 ignored trash。
完整历史测试可能需要按 [archive manifest](../data/data_archive_20260905.json)
恢复对应文件及 sidecars；不要禁用测试或换上新表伪造历史测试通过。
文档更新本身不构成重新通过 CTest 的证据。

## 文档、数据和进度的归属

[物理说明](TOPAS_GPU_Physics_Model.md) 不管理任务状态；
[planning](planning.md) 不重新规定物理；
[scoring](scoring_validation.md) 不复制病例结果；
[results](results.md) 链接冻结证据。
任务状态由 [plan](../plan/README.md) 与 [plan2](../plan2/README.md) 分别控制，
不能因新 benchmark 完成而自动关闭其他工作流。
