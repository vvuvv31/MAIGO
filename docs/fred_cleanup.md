# FRED 相关实现移除记录（2026-09-15）

按用户要求，活动代码中的旧事件库、解析碎裂模型及其经验参数已移除。此前“保留运行依赖”的清理结论已被本次修改取代。当前核末态使用 TOPAS 提取的 CINEL03 数据，统一 EM 和 all-ion elastic 数据不变。

## 移除范围

第一阶段将 67 个已跟踪旧文件及 5 个字节码缓存移入 `trash/fred_cleanup_20260915/`，包括旧配置、FELB 事件库、Table 1 反演、2GR 散射、paper 截面选项及专用提取/比较脚本。逐文件路径和哈希见[清单](fred_cleanup_manifest.json)。

本阶段进一步从活动源码删除：

- `include/carbon/detail/fred_fragmentation_data.hpp`：经验概率、拟合参数、CDF 和相关解析算法。
- `src/detail/sycl_inelastic_device.inc`：解析碎裂采样实现。
- `include/carbon/detail/c12_elastic_h_xs.hpp`：旧 H-only 弹性表路径。
- 主输运中的对应调用、概率上传、专用缓冲区和回读，以及结果结构、日志、JSON、合并和质量检查中的专用字段。
- 仅服务上述模型的测试；step13 通用工具保留，但不再读取已删除字段。

`src`、`include`、`tests`、`tools`、`config` 和 CMake 的活动文本扫描不再包含 FRED/FELB/2GR 标识。退役配置会被拒绝，不会静默切换物理模型。

## 当前运行依赖

原共用功能已从旧模型中解耦：

- `include/carbon/charged_species.hpp` 定义当前 EM 数据包对应的 18 种带电离子及索引，顺序与包内 `species_za` 一致，包括 Be6；不含论文产额或经验碎裂概率。
- `include/carbon/nuclear_collision.hpp` 使用通用指数分布 CDF/逆变换生成有限步内的碰撞距离，供次级非弹性和 all-ion elastic 使用。

原发 post-EM null 分支现在清除碰撞标志、保留轨迹并完成当前 EM 步及记分，不再落入解析碎裂。RT07575 对照仍记录 90 次原发 null candidate，但新源码已无解析模型或对应事件计数。CINEL03 的队列、未输运能量和终止账本保留；仅删除旧解析模型专属账本。核输运要求经过验证的 Schneider/统一水 CINEL03 路径。

## 验证

本地 sm_75 Release 构建 `carbon_mc`、`carbon_tests` 和新增 `nuclear_collision` 测试成功，后者 CTest 通过，覆盖零反应率、有限步截断、指数分布和离子索引。Schneider v2.1、统一 EM 核心及矩数据、all-ion elastic 数据校验通过。

下表为相同输入及随机种子的清理前后 3D 剂量对照，差值均以清理前峰值归一化，**不是相对 TOPAS 的误差或 Gamma 验收**。

| 对照 | 原发数 | 最大绝对差 / 峰值 | RMS 差 / 峰值 | 运行质量 / overflow |
|---|---:|---:|---:|---|
| RT07575，默认弹性关闭 | 3,240,963 | 0.0625665% | 0.00007664% | 通过 / 0 |
| 统一水 200 MeV/u | 50,000 | 0.00005753% | 0.000000259% | 通过 / 0 |
| RT07575，all-ion elastic 开启，research 模式 | 50,000 | 0.0571018% | 0.00003132% | 通过 / 0 |

CT 的小幅变化包括修正 null 分支造成的物理变化。开启弹性的研究验证不等于生产精度认证。未重新执行完整患者 BODY Gamma。之前已确认的两项基线测试失败（`test_run_quality_gate`、`test_strict_config_parsing_and_canonicalization`）未通过放宽质量门槛绕过，因此不声称完整测试集通过。

复现脚本和结果在 `scratch/remove_analytic_model/`，主要结果为 `validation/results.json` 与 `elastic_validation/results.json`；此前基线测试失败证据在 `scratch/fred_cleanup/baseline_tests.json`。

新验证程序为 `scratch/remove_analytic_model/build/carbon_mc`，SHA256：
`839ffe07e182d14889017a881304e1f7b7f5120d9a0882ea124d8c9a260447f7`。
现有生产二进制 `build/oneapi-nvidia-release/carbon_mc` 未覆盖，仍是旧构建，SHA256：
`db4e7345e05bdd941e38076a8e2c51ed6ff5187d9cb31293808eaf97b02b6316`。
默认次级续跑仍为 64 步，调度候选不包含在本次清理提交中。

## 历史与来源

Git 历史、论文引用、历史实验记录以及本地 trash/scratch 中的备份和旧二进制未清除。它们不参与新程序构建；现有生产二进制需要重新构建才会包含本次移除。文献引用和来源说明保留，代码移除本身不构成版权合规认证，也不等于已清理 GitHub 历史。
