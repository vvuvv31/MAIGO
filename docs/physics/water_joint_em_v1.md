> REMOVED from code (2026-09-14): `g4_joint_water_v1` / joint EM support deleted.
> This file is history only; b1 runs on legacy or unified EM.

# C12 均匀水联合电磁候选

状态：已接入主代码的显式研究路径，尚未替换生产默认。原始默认 TOPAS 参数保持不变。

在现有已验证的均匀水 YAML 中增加：

```yaml
run_mode: research
primary_em_model: g4_joint_water_v1
primary_joint_em_data_directory: /absolute/path/to/MAIGO/data/water_joint_em_v1
```

运行前执行 `python3 tools/verify_water_joint_em_data.py`。GPU 后端在加载时再次校验两个 CSV 的固定 SHA256；数据损坏或缺失直接失败。包 manifest 记录提取来源、质量约定、电子阈值、原始样条类型及适用范围。当前 schema 尚不支持替代版本。

配置默认 `primary_em_model: legacy`。回退时移除数据目录配置并选 legacy。旧环境开关 `CARBON_JOINT_EM_DATA` 和 `CARBON_DIAGNOSTIC_PRIMARY_START_DEDX` 在新程序中报错，防止将隔离实验启动方式误用于正式程序。

候选只支持本地 SYCL GPU、原发 C12、密度 1 g/cm³ 均匀水、零能散研究运行。CT、分层材料、异质插入、其他原发以及电子响应叠加不在适用范围。保留 `straggling_scale: 1.0`；不支持分能量涨落缩放。

行为：

- 原生 Geant4 11.3.2 受限 DEDX、range/inverse-range、lambda 样条，加实际输运质量/电荷与离子修正。
- StepFunction(0.1, 0.001 mm) **覆盖**旧 `maximum_step_mm` 和 `maximum_relative_energy_loss`；体素、核与电子碰撞继续裁短步长。启动日志明确报告覆盖。
- 连续受限涨落与显式 δ 电子分开扣能；δ 电子能量局部沉积，没有完整电子空间 tracking。
- 包含已在隔离实验中启用的原发非弹性反应率缓存；弹性仍用局部反应率。次级保持原输运路径。研究配置不依赖特殊编译宏。
- 运行质量报告记录研究模型近似项。`accepted=true` 仅表示现有数值质量检查通过，不能作为生产或 CT 物理验收。

## 集成验证（2026-09-13）

三个组件 CTest 全通过。100/200/300 MeV/u 完整物理 0.5 mm 与纯 EM 0.1 mm 深度网格，共六组，每组 50k，均无 overflow、无候选采样失败。与隔离版本同种子比较：审计计数完全一致；三维剂量最大差异低于峰值的 6.24e-7。该比较证明集成复现，不是新增高统计量物理精度证据。

提交内证据：[集成记录](../../evidence/step-31/joint-em-integration-20260913/README.md)，包括回归、分量分析、二进制 hash 和配置拒绝记录。原始输出位于本地 `scratch/joint_em_integration_20260913`。配置拒绝测试见 `tests/joint_em_config_guard.py`，需要有效水 YAML 与已构建程序；部分无效输入由现有更早的配置检查拒绝。

## 剩余积分剂量差异

按原发来源与其他来源三维计分横向求和，以下差值均除以 TOPAS 总积分剂量：

| MeV/u | 总差异 | 原发来源贡献 | 其他来源贡献 |
|---|---:|---:|---:|
|100|−0.3103%|−0.1039%|−0.2064%|
|200|−0.5182%|−0.1500%|−0.3683%|
|300|−0.6960%|−0.2442%|−0.4518%|

其他来源贡献占缺口约 65%–71%。这只是按来源分解，不等于已证明中性粒子、电子、弹性或某个次级过程是原因；需提高统计量并检查产物/逃逸账本。

## 尚未完成的生产门槛

高统计量固定默认 TOPAS 对照、完整 IDD 和横向 core/halo/profile 验收、旧默认路径回归、吞吐量评估、原生 inverse-range 设备闭合，以及候选专用完整质量门槛仍需完成。CT 另需 Schneider 材料特异受限 EM 数据和现行 v2.1 验收；不得用水表回退。当前不应切换生产默认。
