# 发布兼容代码清理：第五轮（2026-10-07）

本轮清理已确认的元数据命名分支、旧读取器 API、无调用的 MCS 助手、旧网格初始化值和过时注释。配置、数据包和当前物理计算保持一致。

## 改动

- `config.cpp` 的启动检查只使用 `basename.metadata.json`，删除追加 `.bin.metadata.json` 的候选命名及其分支，与当前 rate reader 的默认路径一致。
- `SchneiderRateTable::from_binary()` 固定接受 `SCHNRATE` v3，删除 `expected_magic`、`expected_version` 参数及对应通用判断；保留实际使用的显式 metadata 路径参数和完整身份/网格/domain 校验。
- 删除无调用的 `schneider_section_radiation_length_g_per_cm2()`，保留主/次级共用且仍有调用的 MCS 选择器。
- 目标采样器和次级设备上下文的旧 860 节点/0.5 MeV/u 起点默认值改为当前 921 节点/0.1 MeV/u；空次级设备上下文的 projectile 数设为 0。有效运行继续由必需的数据包填入网格及注册表数量。
- 修正旧弹性包、多格式 API、860 节点布局、旧配置 schema、已删除 ion-physics manifest 和 SHA/path pinning 注释。Schema 1 是内部 provenance 常量，当前 YAML 没有这个配置入口。
- 更新 README、AGENTS 与输入契约检查；IEC 水模体坐标、DICOM LPS、随机流、铜碎片物理近似及有效 MCS/记分路径保留。

## 验证

- 最终构建成功（`release5_final_build_20261007.log`）；现有编译警告仍在。
- 15 项输入契约检查通过（`release5_final_inputs_20261007.json`），包括新增的旧 magic、旧版本和仅旧名称 metadata 的拒绝检查。
- 60 项接口检查通过（`release5_interfaces_20261007.json`）。
- 4 项核反应包契约检查通过（`release5_contracts_20261007.json`）。
- 八预设匹配 GPU 回归均通过（`validation_release5_20261007/summary.json`）：每个预设每个臂连续 2 批，每批 10,000 histories；源 CSV/种子和离散 EM/电子/光子/Urban/核反应审计一致，能量闭合、覆盖、完成检查通过，所有队列零溢出。
- 最大剂量相对 L1 差异 `2.83042789737e-07`（阈值 `1e-5`），16 个候选批次最大物理相对能量残差 `7.26624245903e-06`（阈值 `1e-4`）。
- 源码清单 126 个文件及输入清单核对通过；目标旧符号、旧候选路径和 860 网格默认值在对应运行源码中无残留。

冻结对照位于 `/tmp/maigo_cleanup5_before_20261007/`，包含未修改的八个配置、源码和二进制。对照/候选 SHA256 见 `release5_control_20261007.json`；此前历史验证记录保持原样。

本次是匹配 GPU 实现检查，不是新 TOPAS 剂量验收或性能测量。碳 minibeam 剂量精度验收仍待完成。
