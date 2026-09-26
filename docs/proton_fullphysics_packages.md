# Proton 主束 full physics 包（2026-09-26）

本次主束提取范围：**0.1–250 MeV，非 minibeam，纯水及 25 个 Schneider 材料分区**。
TOPAS 4.2.p3 / Geant4 11.3.2。EM 水参考为 `Water_75eV`，未用比例系数校准。

## 使用

入口：`config/proton_water_fullphysics.yaml`。默认质子 150 MeV、零能散、10,000 histories，450 mm 水模体。

```bash
LD_LIBRARY_PATH=/home/wuwei/sycl_workspace/llvm/build/install/lib:${LD_LIBRARY_PATH:-} \
  build/oneapi-nvidia-primary/carbon_mc --config config/proton_water_fullphysics.yaml
```

改 `initial_energy_MeV` 可选择 0.1–250 MeV；默认零能散避免最高能量端越出范围。
已有固定二进制网格（rate 至 460.1 MeV/u、stopping 至 430.11 MeV/u）保持兼容，
**固定网格上限不代表新主束包支持到该能量**。核末态提取至 251 MeV，作为 250 MeV 插值的保护节点。
启动检查读取 bundle 的 `source_energy_scope_MeV`，拒绝名义源能量越界。

数据目录：`data/proton_therapy_20260926/`。完整文件哈希见目录内 `SHA256SUMS`；
原始提取与构建日志位于 `/mnt/sda/wuwei/proton_fullphysics_20260926/`。
YAML 中路径为本机绝对路径；迁移时一并复制包目录并更新 YAML 路径。

## 新提取与共享数据

| 数据 | 本次处理 |
|---|---|
| 主束 CINEL03 相关核末态 | 新提取；242,027 个 proton 事件；实际过程 `protonInelastic`，模型 `Binary Cascade` |
| 主束 SCHNRATE v3 | 新提取 25 分区 × 13 元素靶截面，与末态支持域匹配 |
| 主束 SCHNSTOP v1 | 新提取 proton，25 分区 × 4,302 节点，元数据 Z/A=(1,1) |
| 水主束 stopping CSV | 新提取 `Water_75eV`，4,001 节点 |
| all-ion elastic bank | 新提取 proton 弹性行，17 个其他物种行从已固定哈希的 TOPAS 包保留 |
| recoil stopping | 重新提取全部反冲物种，水采用 `Water_75eV` |
| unified EM、δ moments | 共享既有 18 物种包，包含真实 proton 表；核对原固定哈希 |
| 次级 nuclear rate / CINEL03 | 共享既有 14-projectile 包，保持原始元数据与哈希 |
| 次级水/Schneider stopping | 保留包含 proton 的多物种表；Schneider 附件供 CT 配置使用 |

共享是复用原本就包含该物种的数据，未将 C12 数据改名成 proton。
`shared/manifest.json` 保留共享输入来源，新的 elastic metadata 记录被替换的 proton 原始数据和父包 SHA。

## 低能提取与零通道

自然输运首轮：13 元素 × 13 束能 × 10,000 histories，共 169 个任务、178,729 个事件。
审计发现 66 个低能非零截面点没有末态样本；为这些点单独做条件末态提取。

`MAIGO_CINEL_FINAL_STATE_XS_FACTOR` 仅用于此项提取，通过 Geant4 的
`MultiplyCrossSectionBy` 增强 primary proton 在停止前发生核反应的概率。
Geant4 原有靶同位素选择与 Binary Cascade 末态模型保持启用，记录实际碰撞能量与完整相关产物。
该开关要求 `CARBON_CINEL02_PRIMARY_ONLY=true`，会写入 TOPAS 日志和 raw contract。
**这些增强任务的 exposure 不能用于计算物理发生率或剂量**；运行时 rates 来自另一组完全未增强的 TOPAS 截面导出。

补提取共 67 个任务，63,298 个事件。最终 `coverage.json` 报告：

- 大于 5 MeV 的内部能量空档：0。
- 0.1–250 MeV 内截面非零但末态域未覆盖的网格点：0。
- p+H 在本次范围内的 TOPAS 非弹性截面为零；其 channel 明确设 `has_support=false`，全部存储发生率为零，未填充虚构事件。

数据编译审计包括 raw CRC、源/靶身份、实际过程与模型、单位权重、相关方向、产物计数、能量字段及既有宽松核事件能量闭合界限。
这不是逐事件精确核质量 Q 值闭合验证，也不是剂量验收。

## 复现工具

- `extensions/tools/primary_source/campaign.py`：生成独立任务、执行 TOPAS、记录输入/二进制/raw 哈希。
- `extensions/tools/primary_source/compile.py`：校验 raw、生成 CINPKG04、SCHNRATE v3、channels 与 bundle；未覆盖的正截面点会阻止最终编译。
- `extensions/tools/schneider/compile_schneider_stopping.py`：支持 `--projectile-z 1 --projectile-a 1 --prefix proton_`。
- `extensions/tools/primary_source/assemble.py`：组装 elastic bank、反冲与水 stopping、可运行 YAML。

本次所有 TOPAS 任务经本地 Slurm 执行。原始输入、任务状态、TOPAS 可执行文件和扩展源码快照均保留在工作目录。

## 验收状态与范围

二进制已通过当前 C++ 加载器的数据契约检查；YAML 已通过不启动输运的预加载。
GPU 程序使用关闭 minibeam 的 FP32 构建（FP64 scorer 关闭）。

此处 full physics 指当前 MAIGO 支持的统一 EM/涨落、MCS、核非弹性及次级输运、核弹性和反冲组合。
默认 MCS 是 **Highland**；TOPAS 日志中的 proton MCS 为 **WentzelVIUni**，本次未实现该模型的 GPU 复刻。
次级 registry 外核反应仍遵循 `em_only`，中性粒子处理沿用现有实现。
数据提取阶段未进行端到端剂量对照；后续已完成 Urban full physics 水中对照，见下节。尚未确立临床剂量精度验收。

### Urban v2 可选入口

[`proton_water_fullphysics_urban_v2.yaml`](../config/proton_water_fullphysics_urban_v2.yaml)
在同一套质子 full physics 包上启用通用 Urban v2，包含主束、带电次级和反冲。
数据范围、物理水边界及与默认 TOPAS Wentzel 的区别见 [proton_urban_v2.md](proton_urban_v2.md)。
该配置现已完成 70/120/170/220 MeV 的 GPU/TOPAS 对照（每个引擎每个能量 60 万质子）；[结果、图表与物理范围](../evidence/proton_urban_full_20260926/README_zh.md)。束芯与 RMS 尾部的差异分别报告，不能以运行质量通过代替剂量一致性验收。

### 500 万质子统计复测

已完成每个能量、每个引擎 500 万质子的 Urban full physics 复测；包含逐点 IDD 误差、束芯与 RMS、相对 60 万的统计收敛对照。
本轮还修复了 FP32 随机数上端点导致零核碰撞距离的问题，并统一重跑 GPU。
[报告与范围说明](../evidence/proton_urban_full_5m_20260926/README_zh.md)。
