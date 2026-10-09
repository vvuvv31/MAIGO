# RT07575 head carbon minibeam：离子弹性参考配置修复

## 已修复的确定问题

2026-10-07 夜间 TOPAS 使用七个物理模块，包含 `g4h-elastic_HP`，但没有 `G4IonElasticPhysics`。原始 spot 1507 日志的 `Hadronic Processes for GenericIon` 只有 `ionInelastic` 和 `RadioactiveDecay`。GPU 当时已启用主碳离子在铜及 CT 中的核弹性。因此原比较存在确定的物理过程配置不一致。

新增 `extensions/topas/common/CarbonIonElasticPhysics.{hh,cc}`，仅包装官方 `G4IonElasticPhysics`，在 `ConstructProcess()` 后检查 GenericIon 的 `G4HadronElasticProcess`，并输出实际注册的过程名称。它不修改截面、模型或 GPU 输运，也不自动激活扩展：TOPAS 输入必须显式增加模块并更新元素数量。

`scripts/check_topas_carbon_reference.py` 检查上述实际注册标记或 Geant4 的 GenericIon hadronic process summary。原参考检查失败；修复构建实跑输出 `GenericIon process=ionElastic registered=true`，检查通过。它只检查弹性注册，不是全物理审核或剂量验收。

使用方法见 `extensions/README.md` 的“碳离子 TOPAS 剂量参考的核弹性过程”。构建脚本已有 common sources staging，不需要新增构建模式。

## 对照结果：只修复了部分差异

对 spot 1507、1629、1382，保留原本地 TOPAS 的 histories、seed、源扩展、束斑、CT、准直器和单次物理剂量归一化；仅增加官方离子弹性模块。用于运行的既有独立 pencil-beam TOPAS 构建包含相同 PBS 源文件，逐文件字节比对通过。GPU 使用原夜间结果，没有调参、缩放或平移。三个对照共完成约七分钟（本地 Slurm 25250，100 线程，退出 0）。注册防护的新构建另完成一次单 history 功能检查（25251，退出 0），此功能检查不用于剂量或性能结论。

患者 y=123.2 mm、GTV 内每个 spot 的平均剂量差：

|spot|旧 GPU−TOPAS (Gy)|补齐离子弹性后 GPU−TOPAS (Gy)|有符号负差缩小|
|---|---:|---:|---:|
|1507|−0.019729|−0.008184|58.5%|
|1629|−0.017432|−0.015750|9.7%|
|1382|−0.012144|−0.014283|−17.6%，负差增大|

三个束斑合计，固定原三束斑剂量在 GTV 中 ≥其 GTV 最大值 10% 的 9,252 个参考体素，使用 local 3%/0.2 mm、search 0.02 mm：

|参考|通过率|
|---|---:|
|原 TOPAS，无 GenericIon 核弹性|83.5171%|
|补齐 GenericIon 核弹性|83.6468%|

这不是全计划 gamma，也不直接与此前全计划 87.5449% 比较。当前结果没有证明条带已消失。对照同 seed 不意味着过程改变后仍逐 history 配对；GPU 空间协方差未测量，不能把表中变化直接解释为精确模型误差或显著性。原生 TOPAS parallel-world 导航漏分日志仍保留：三例旧/新 unscored energy 分别为 0.09143/0.0378986、0.0697366/0.0184072、0.0544644/0.0987824 MeV（以结果 JSON 为准）；它们不是被人为补偿进剂量的能量。

## 剩余问题和完成边界

确认漏掉离子弹性，不等于找到了条带全部根因。剩余的峰低、谷高及随深度变化，仍需定位到铜后束形/成分与 CT 内横向输运。当前没有充分证据支持修改 `g4_ion_delta`、Urban 幅度、狭缝宽度或 spot weight，故未进行这些物理调整。

完整的匹配物理参考需要所有 400 个 spot 用补齐弹性的 TOPAS 重新模拟；旧参考剂量不能通过后处理补上核弹性。此次没有启动全计划重算，也不能宣布 carbon minibeam 剂量接受。只改了外部参考扩展、检查器和文档，生产 GPU 物理和八个预设保持原样。

## 证据位置

`out/rt07575_head_carbon_overnight_20261007/strip_fix/`：

- `ionelastic_results.json`、`ionelastic_comparison.png`、`analyze_ionelastic.py`：三个束斑的剂量和 gamma 对照。
- `ionelastic_provenance.json`：运行二进制摘要及 PBS 字节一致性。
- `topas_ionelastic/spot_*/`：输入、完整日志和原生 dose_stats 输出。
- `build_reference/`、`process_guard/audit.json`：包含注册防护的新扩展编译、链接及实际过程检查。
- `original_process_audit.json`：原参考的缺失过程检查。

前期铜出口评分尝试未作为证据：TOPAS 第一版 Surface 参数错误；第二版在取消前已完成，但仅评分 Aperture 母体出口，没有覆盖狭缝内的直穿主束，不能作为完整出口参考。离线 GPU 输出没有得到有效出口记录（生产代码中该记录分配已禁用，未恢复诊断模式），虽其能量/完成/队列审计通过，也不据此推断出口束形。初版 gamma 检查因 NaN reference 的默认归一化失败，最终显式指定参考最大值后重算并检查了完整参考 mask。所有这些无效诊断保留在独立目录，没有覆盖原夜间结果。
