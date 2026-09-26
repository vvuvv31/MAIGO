# Proton source Urban v2 接入

入口：[`config/proton_water_fullphysics_urban_v2.yaml`](../config/proton_water_fullphysics_urban_v2.yaml)。
在已有 0.1–250 MeV proton full physics 包上启用通用 Urban v2，默认 150 MeV、Water_75eV、非 minibeam。

```bash
LD_LIBRARY_PATH=/home/wuwei/sycl_workspace/llvm/build/install/lib:${LD_LIBRARY_PATH:-} \
  build/oneapi-nvidia-primary/carbon_mc \
  --config config/proton_water_fullphysics_urban_v2.yaml
```

## 接入内容

通用主束输运已有 `urban_device.projectile(primary_atomic_number, primary_mass_number)` 路由，
并将实际动能和质子质量交给 Urban 步长、角度及位移计算。本次补齐 proton 的包与配置入口，无需新增输运分支。

```yaml
multiple_scattering_model: urban_v2
urban_mcs_package_file: /path/to/urban_mcs_v2.bin
urban_mcs_package_sha256: 493015506d0168efc16afae6b81441010d7937e899d3e88196c45a5b8c511eb9
urban_water_half_width_mm: 2000
secondary_local_deposit_cutoff_MeV: 0.1
```

- 开关作用于主束 proton、被输运的带电次级粒子及弹性反冲；不是仅主束开关。
- `urban_water_half_width_mm` 定义水母体的物理半宽；当前为 2000 mm，计分 ROI 仍为原配置的半宽 64 mm。
- 主束和次级终止动能均为 0.1 MeV。Urban 参考要求至少 0.05 MeV。
- 水密度为 1 g/cm³，production cut 为 0.05 mm。该 cut 不是 `maximum_step_mm`。
- 原有核反应、EM、δ moments 和反冲包继续由基础 full physics YAML 的路径与哈希指定。

## 数据范围与来源

包复制到 `data/proton_urban_v2_20260926/urban_mcs_v2.bin`，约 133 MiB。
该包来自既有全带电离子 Urban 参考，包含 52 个物种，明确含 `(Z,A)=(1,1)`。
proton 水记录的 factor 动能范围约 0.05–6500 MeV，MFP 范围约 1 eV–6500 MeV，覆盖本次主束能区。
提取器显式移除了默认 hadron MCS，注册 `G4UrbanMscModel`，因此该数据是 proton Urban 数据。

水使用 `Water_75eV`。CT 部分仅支持包内 4 个准确材料/密度对（section 0、1、8、20）；
**不是覆盖全部 25 Schneider 分区及任意密度的 Urban 包**。当前新增入口限定纯水。
来源与 SHA 记录在同目录的 `manifest.json`、`reference_manifest.json`、`SHA256SUMS`。

## 状态

已完成 70、120、170、220 MeV 的非 minibeam Water_75eV **EM-only** 对照，每个引擎、每个能量 3×200,000 个质子。
GPU 在本机 RTX 2080 Ti 直接运行（FP32 dose），TOPAS 经本地 Slurm 运行。

TOPAS opt4 默认 proton MCS 是 `WentzelVIUni`；本次新增显式模块
`AllChargedUrbanPhysics`，在 opt4 后替换 hadron MSC 为 Urban，日志已核实质子 `UrbanMsc`。
GPU 的 12 个批次均通过运行质量检查，Unified EM 失败数与队列溢出均为零。

- GPU−TOPAS R80：依次 +0.027、+0.049、+0.074、+0.076 mm。
- IDD L1 相对差：依次 0.441%、0.248%、0.186%、0.146%。
- 2 mm 至 90% R80 内，x/y 高斯束芯 sigma 平均相对差：依次 −0.238%、−0.259%、−0.366%、−0.453%。

[完整报告、图表、实际配置与统计定义](../evidence/proton_urban_em_20260926/README_zh.md)。
两端的 delta 电子处理、低能终止等仍有差别，详见报告。该结果不代表 proton full physics 或异质 CT 已完成剂量验收。

## Full physics 对照（2026-09-26）

已进一步完成相同 70/120/170/220 MeV、每个引擎每个能量 3×200,000 histories 的 full physics 对照。
GPU 直接运行，TOPAS 本地 Slurm；打开核非弹性、核弹性、当前支持的带电次级和反冲输运，继续使用 Urban。

IDD L1 差依次为 0.462%、0.516%、0.753%、0.606%；束芯高斯 σ 平均偏差依次 −0.202%、−0.307%、−0.527%、−0.573%。
RMS 的偏差明显大于束芯，且有随深度正负抵消；不能仅用束芯 σ 判断核反应尾部已经一致。
[完整 full physics 报告、RMS 独立图和核能量审计](../evidence/proton_urban_full_20260926/README_zh.md)。

GPU 的中性核产物仍采用未输运能量记账，registry 外核反应为 EM-only，级联代数有上限；因此当前 full physics 组合不等同于 TOPAS 所有物理过程均已实现。

## 500 万质子统计复测（2026-09-26）

70/120/170/220 MeV 各完成每个引擎 5×100 万质子，full physics + Urban；GPU 直接运行，TOPAS 本地 Slurm。
IDD 采用同深度 TOPAS 为分母的逐点相对误差。统计标准误差相对原 60 万降低 2.67–2.94 倍。

| 能量 MeV | IDD 平均绝对逐点误差（入口至 R80） | 束芯 σ 平均相对差 | RMS 平均绝对相对差 |
|---|---|---|---|
| 70 | 0.268% | -0.059% | 6.750% |
| 120 | 0.365% | -0.204% | 6.033% |
| 170 | 0.668% | -0.374% | 5.537% |
| 220 | 0.557% | -0.576% | 2.279% |

σ 的统计区间为 2 mm 至 TOPAS R80 的 90%。尾部偏差在增加统计量后仍存在。
本轮发现 `rng::uniform01` 的 FP32 上端点舍入到 1，可产生零核碰撞距离；已修复端点并用原种子统一重跑全部 20 个 GPU 批次。
所有最终批次通过审计，无 EM 失败或队列溢出；失败及修复前结果单独归档。
[500 万完整报告、逐点误差与统计对比图](../evidence/proton_urban_full_5m_20260926/README_zh.md)。

## Copper minibeam extension (2026-09-26)

The research minibeam source route is now available with a separate G4_Cu Urban record and proton copper packages. See [proton_minibeam.md](proton_minibeam.md) for configuration, benchmark geometry and the remaining copper-secondary/air/neutral transport limitations. The original water-only package does not contain copper.
